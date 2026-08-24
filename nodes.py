"""ComfyUI node orchestration for long H3 reference video with fixed audio."""
from __future__ import annotations

import torch
import torchaudio

import comfy.nested_tensor
import comfy.samplers

from .fixed_audio import (
    _FixedAudioWindowSampler,
    _align_frame_count,
    _audio_step_at_frame,
    _av_parts,
    _plan_windows,
    _slice_audio_window,
    _slice_video_window,
    _unpack_official_conditioning,
    _validate_audio,
    _video_latent_steps,
)
from .h3_motion_context import apply_motion_context


class MinimaxH3RefSampler:
    """Single-timeline H3 reference director with sigma-fixed target audio."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            **{f"ref_image_{i}": ("IMAGE",) for i in range(9)},
            **{f"ref_video_{i}": ("IMAGE",) for i in range(3)},
            **{f"ref_audio_{i}": ("AUDIO",) for i in range(3)},
            **{f"ref_video_audio_{i}": ("AUDIO",) for i in range(3)},
        }
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 864, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 480, "min": 32, "max": 8192, "step": 32}),
                "frames": ("INT", {"default": 362, "min": 5, "max": 999999, "step": 17}),
                "window_frames": ("INT", {"default": 362, "min": 22, "max": 362, "step": 17}),
                "context_frames": (["5", "22", "39", "56"], {"default": "22"}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                }),
                "steps": ("INT", {"default": 8, "min": 1, "max": 1000}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "beta"}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            },
            "optional": {"target_audio": ("AUDIO",), **optional},
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "run"
    CATEGORY = "sampling/minimax"

    def run(self, model, clip, video_vae, audio_vae, prompt, width, height,
            frames, window_frames, context_frames, ref_image_size, seed, steps,
            scheduler, shift_video, shift_audio, target_audio=None, **refs):
        try:
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        except ImportError as exc:
            raise RuntimeError(
                "Minimax H3 Ref Sampler requires the official ComfyUI "
                "MiniMax H3 nodes"
            ) from exc

        full_clean_audio = None
        if target_audio is not None:
            waveform, sample_rate = _validate_audio(target_audio)
            if waveform.shape[1] != 1:
                waveform = waveform.mean(dim=1, keepdim=True)
            vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
            encode_waveform = waveform
            if sample_rate != vae_rate:
                encode_waveform = torchaudio.functional.resample(
                    waveform, sample_rate, vae_rate
                )
            full_clean_audio = audio_vae.encode(encode_waveform.movedim(1, -1))
            if (full_clean_audio.ndim != 4
                    or full_clean_audio.shape[0] != 1
                    or full_clean_audio.shape[1:3] != (32, 2)):
                raise ValueError(
                    "H3 Audio VAE must encode target_audio as [1,32,2,T], got "
                    f"{tuple(full_clean_audio.shape)}"
                )
            full_clean_audio = full_clean_audio.detach().to(
                device="cpu", dtype=torch.float32
            )
        total_frames = _align_frame_count(int(frames))
        window_frames = _align_frame_count(min(362, int(window_frames)))
        context_frames = int(context_frames)
        windows = _plan_windows(total_frames, window_frames, context_frames)

        static_images = {
            key: value for key, value in refs.items()
            if key.startswith("ref_image_") and value is not None
        } or None
        source_videos = {
            key: value for key, value in refs.items()
            if key.startswith("ref_video_") and not key.startswith("ref_video_audio_")
            and value is not None
        }
        source_audios = {
            key: value for key, value in refs.items()
            if key.startswith("ref_audio_") and value is not None
        }
        source_video_audios = {
            key: value for key, value in refs.items()
            if key.startswith("ref_video_audio_") and value is not None
        }

        sampler = _FixedAudioWindowSampler()
        video_parts, audio_parts = [], []
        stitched_audio_t = 0
        global_audio_noise = None
        output_template = None
        previous_sampled = None
        context_t = _video_latent_steps(context_frames)

        for index, (start_frame, sample_frames) in enumerate(windows):
            ref_videos = {
                key: _slice_video_window(value, start_frame, sample_frames)
                for key, value in source_videos.items()
            } or None
            ref_audios = {
                key: _slice_audio_window(value, start_frame, sample_frames)
                for key, value in source_audios.items()
            } or None
            ref_video_audios = {
                key: _slice_audio_window(value, start_frame, sample_frames)
                for key, value in source_video_audios.items()
            } or None
            conditioned = MiniMaxH3ReferenceToVideo.execute(
                clip, video_vae, audio_vae, prompt,
                int(width), int(height), int(sample_frames), ref_image_size,
                ref_images=static_images,
                ref_videos=ref_videos,
                ref_video_audios=ref_video_audios,
                ref_audios=ref_audios,
            )
            positive, latent = _unpack_official_conditioning(conditioned)
            if previous_sampled is not None:
                positive, pinned_frames = apply_motion_context(
                    positive,
                    latent,
                    context_length=context_frames,
                    context_latent=previous_sampled,
                    continue_audio=full_clean_audio is None,
                    audio_context_length=context_frames,
                )
                if pinned_frames != context_frames:
                    raise RuntimeError(
                        "Motion Context handoff does not match the planned overlap: "
                        f"pinned={pinned_frames}, expected={context_frames}"
                    )
            if output_template is None:
                output_template = dict(latent)
            _, audio_template = _av_parts(latent)
            window_audio_t = int(audio_template.shape[-1])
            wanted_audio_end_t = _audio_step_at_frame(
                start_frame + sample_frames
            )
            audio_window_start_t = wanted_audio_end_t - window_audio_t
            if audio_window_start_t < 0:
                raise RuntimeError("H3 audio window starts before the timeline")
            append_audio_t = wanted_audio_end_t - stitched_audio_t
            if not 0 < append_audio_t <= window_audio_t:
                raise ValueError(
                    "audio window cannot cover its absolute timeline range: "
                    f"need={append_audio_t}, available={window_audio_t}"
                )
            if full_clean_audio is not None and global_audio_noise is None:
                # One absolute noise timeline keeps overlapping target-audio
                # positions identical across independently seeded video windows.
                noise_t = max(
                    wanted_audio_end_t,
                    _audio_step_at_frame(total_frames) + 4,
                )
                generator = torch.Generator(device="cpu").manual_seed(int(seed))
                global_audio_noise = torch.randn(
                    (*audio_template.shape[:-1], noise_t),
                    generator=generator,
                    dtype=torch.float32,
                )
            audio_noise = None
            clean_audio = None
            if full_clean_audio is not None:
                audio_noise = global_audio_noise[
                    ..., audio_window_start_t:wanted_audio_end_t
                ]
                if audio_noise.shape[-1] != window_audio_t:
                    raise RuntimeError("global fixed-audio noise timeline is too short")
                clean_audio = full_clean_audio[
                    ..., audio_window_start_t:wanted_audio_end_t
                ]
            sampled = sampler.sample(
                model, positive, latent, clean_audio,
                (int(seed) + index) & 0xFFFFFFFFFFFFFFFF,
                steps, scheduler, shift_video, shift_audio,
                audio_noise,
            )
            window_video, window_audio = _av_parts(sampled)
            if window_audio.shape[-1] != window_audio_t:
                raise RuntimeError("sampler changed the H3 audio window length")
            if index == 0:
                video_parts.append(window_video)
            else:
                video_parts.append(window_video[:, :, context_t:])
            audio_parts.append(window_audio[..., -append_audio_t:])
            stitched_audio_t = wanted_audio_end_t
            if index + 1 < len(windows):
                # Motion Context extracts the phase-aligned AV tail and injects
                # it into the next window's conditioning. The next target
                # latent itself stays untouched.
                previous_sampled = sampled

        if output_template is None:
            raise RuntimeError("long-video planner produced no windows")
        out_video = torch.cat(video_parts, dim=2)
        wanted_video_t = _video_latent_steps(total_frames)
        total_audio_t = max(1, _audio_step_at_frame(total_frames))
        out_audio = torch.cat(audio_parts, dim=-1)
        if out_video.shape[2] != wanted_video_t:
            raise RuntimeError(
                "stitched video latent length mismatch: "
                f"expected={wanted_video_t}, actual={out_video.shape[2]}"
            )
        if out_audio.shape[-1] != total_audio_t:
            raise RuntimeError(
                "stitched audio latent length mismatch: "
                f"expected={total_audio_t}, actual={out_audio.shape[-1]}"
            )
        output = output_template
        output.pop("noise_mask", None)
        output["samples"] = comfy.nested_tensor.NestedTensor((
            out_video, out_audio,
        ))
        return (output,)


NODE_CLASS_MAPPINGS = {
    "MinimaxH3RefSampler": MinimaxH3RefSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimaxH3RefSampler": "Minimax H3 Ref Sampler",
}
