"""Single-node, in-loop MiniMax H3 masked music-video generation."""
from __future__ import annotations

import torch

import comfy.nested_tensor
import comfy.samplers
import comfy.utils

from .music_video import (
    MaskedAVWindowSampler,
    PixelCrossfadeAssembler,
    _align_frame_count,
    _audio_step_at_frame,
    _av_parts,
    _plan_windows,
    _prepare_masked_target,
    _prepare_master_waveform,
    _slice_audio_window,
    _slice_video_window,
    _unpack_official_conditioning,
    _video_latent_steps,
    ensure_h3_av_mask_support,
)


class H3TalkingSampler:
    """Generate a long masked-context music video on one master timeline."""

    @classmethod
    def INPUT_TYPES(cls):
        default_sampler = (
            "res_multistep"
            if "res_multistep" in comfy.samplers.SAMPLER_NAMES
            else comfy.samplers.SAMPLER_NAMES[0]
        )
        default_scheduler = (
            "simple"
            if "simple" in comfy.samplers.SCHEDULER_NAMES
            else comfy.samplers.SCHEDULER_NAMES[0]
        )
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
                "master_audio": ("AUDIO",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 864, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 480, "min": 32, "max": 8192, "step": 32}),
                "frames": ("INT", {"default": 872, "min": 5, "max": 999999, "step": 17}),
                "window_frames": ("INT", {"default": 362, "min": 22, "max": 362, "step": 17}),
                "context_frames": (["5", "22", "39", "56", "73", "90"], {"default": "39"}),
                "video_crossfade_frames": ("INT", {"default": 39, "min": 0, "max": 90, "step": 1}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                }),
                "steps": ("INT", {"default": 25, "min": 1, "max": 1000}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default": default_sampler}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": default_scheduler}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "AUDIO")
    RETURN_NAMES = ("samples", "images", "master_audio")
    FUNCTION = "run"
    CATEGORY = "sampling/minimax"

    def run(self, model, clip, video_vae, audio_vae, master_audio, prompt,
            width, height, frames, window_frames, context_frames,
            video_crossfade_frames, ref_image_size, seed, steps,
            sampler_name, scheduler, shift_video, shift_audio, **refs):
        try:
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        except ImportError as exc:
            raise RuntimeError(
                "Minimax H3 Ref Sampler requires the official ComfyUI H3 nodes"
            ) from exc

        ensure_h3_av_mask_support()
        master_waveform = _prepare_master_waveform(audio_vae, master_audio)
        total_frames = _align_frame_count(int(frames))
        window_frames = _align_frame_count(min(362, int(window_frames)))
        context_frames = int(context_frames)
        windows = _plan_windows(total_frames, window_frames, context_frames)
        overlap = min(max(0, int(video_crossfade_frames)), context_frames)

        static_images = {
            key: value for key, value in refs.items()
            if key.startswith("ref_image_") and value is not None
        } or None
        source_videos = {
            key: value for key, value in refs.items()
            if key.startswith("ref_video_")
            and not key.startswith("ref_video_audio_")
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

        sampler = MaskedAVWindowSampler()
        assembler = PixelCrossfadeAssembler(overlap)
        video_parts: list[torch.Tensor] = []
        audio_parts: list[torch.Tensor] = []
        stitched_audio_t = 0
        output_template = None
        previous_sampled = None
        context_t = _video_latent_steps(context_frames)
        sampling_work = max(1, int(steps))
        work_per_window = sampling_work + 1
        total_work = len(windows) * work_per_window
        progress = comfy.utils.ProgressBar(total_work)

        for window_index, (start_frame, sample_frames) in enumerate(windows):
            def progress_callback(step, _x0, _x, total_steps):
                completed = round(
                    (int(step) + 1) / max(1, int(total_steps)) * sampling_work
                )
                progress.update_absolute(
                    window_index * work_per_window
                    + min(sampling_work, completed),
                    total_work,
                )

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
            positive, target = _unpack_official_conditioning(conditioned)
            prepared = _prepare_masked_target(
                target, audio_vae, master_waveform,
                start_frame, sample_frames, context_frames, previous_sampled,
            )
            sampled = sampler.sample(
                model, positive, prepared,
                int(seed),
                steps, sampler_name, scheduler, shift_video, shift_audio,
                callback=progress_callback,
            )
            window_video, window_audio = _av_parts(sampled)
            decoded = video_vae.decode(window_video)
            # The first clip also reserves its final overlap for the next seam.
            assembler.add(decoded, context_frames)

            window_video_cpu = window_video.detach().to("cpu")
            window_audio_cpu = window_audio.detach().to("cpu")
            if output_template is None:
                output_template = dict(target)
                video_parts.append(window_video_cpu)
            else:
                video_parts.append(window_video_cpu[:, :, context_t:])

            wanted_audio_end = _audio_step_at_frame(start_frame + sample_frames)
            append_audio_t = wanted_audio_end - stitched_audio_t
            if not 0 < append_audio_t <= window_audio_cpu.shape[-1]:
                raise RuntimeError("audio window cannot cover its absolute timeline")
            audio_parts.append(window_audio_cpu[..., -append_audio_t:])
            stitched_audio_t = wanted_audio_end

            previous_sampled = dict(sampled)
            previous_sampled.pop("noise_mask", None)
            previous_sampled["samples"] = comfy.nested_tensor.NestedTensor((
                window_video_cpu, window_audio_cpu,
            ))
            # Reserve one unit per window for VAE decode and seam assembly, so
            # the frontend reaches 100% only after the window is fully handled.
            progress.update_absolute(
                (window_index + 1) * work_per_window,
                total_work,
            )

        if output_template is None:
            raise RuntimeError("music-video planner produced no windows")

        out_video = torch.cat(video_parts, dim=2)
        out_audio = torch.cat(audio_parts, dim=-1)
        wanted_video_t = _video_latent_steps(total_frames)
        wanted_audio_t = max(1, _audio_step_at_frame(total_frames))
        if out_video.shape[2] != wanted_video_t:
            raise RuntimeError(
                f"stitched video latent mismatch: {out_video.shape[2]} != {wanted_video_t}"
            )
        if out_audio.shape[-1] != wanted_audio_t:
            raise RuntimeError(
                f"stitched audio latent mismatch: {out_audio.shape[-1]} != {wanted_audio_t}"
            )

        images = assembler.finish()
        if images.shape[0] != total_frames:
            raise RuntimeError(
                f"assembled image count mismatch: {images.shape[0]} != {total_frames}"
            )

        output = output_template
        output.pop("noise_mask", None)
        output["samples"] = comfy.nested_tensor.NestedTensor((out_video, out_audio))
        # The AUDIO output bypasses H3 Audio-VAE decode: this is the untouched
        # authoritative soundtrack, matching MultiRef Music Video semantics.
        return output, images, master_audio


NODE_CLASS_MAPPINGS = {"H3TalkingSampler": H3TalkingSampler}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3TalkingSampler": "H3 Talking Sampler",
}
