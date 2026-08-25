"""In-loop MiniMax H3 music-video generation with a fixed master soundtrack.

The AV-mask compatibility modules in this project are derived from
seitanism/ComfyUI-H3-Motion-Context-MultiRef under GPL-3.0.  This orchestration
keeps the same masked-target semantics while running every clip inside one
ComfyUI node.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torchaudio

import comfy.model_sampling
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils

from .h3_mask_compat import ensure_h3_mask_compat
from .h3_mask_payload_compat import ensure_av_mask_payload_compat

H3_VIDEO_FPS = 24
H3_AUDIO_LATENT_FPS = 40
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def ensure_h3_av_mask_support() -> None:
    """Enable only mask capabilities missing from the live ComfyUI build."""
    ensure_h3_mask_compat()
    ensure_av_mask_payload_compat()


def _audio_step_at_frame(frame_index: int) -> int:
    return round(int(frame_index) / H3_VIDEO_FPS * H3_AUDIO_LATENT_FPS)


def _sample_at_frame(frame_index: int, sample_rate: int) -> int:
    return round(int(frame_index) / H3_VIDEO_FPS * int(sample_rate))


def _video_latent_steps(frame_count: int) -> int:
    if frame_count < 5 or frame_count % 17 != 5:
        raise ValueError("H3 frame count must be 5 + 17*n")
    return 2 if frame_count == 5 else ((frame_count - 5) // 17) * 5 + 2


def _pixel_frames(latent_steps: int) -> int:
    return sum(FRAME_PER_TOKEN[index % 5] for index in range(int(latent_steps)))


def _align_frame_count(frame_count: int) -> int:
    frame_count = max(5, int(frame_count))
    return frame_count + ((5 - frame_count) % 17)


def _plan_windows(total_frames: int, window_frames: int,
                  context_frames: int) -> list[tuple[int, int]]:
    """Return absolute ``(start, raw_length)`` H3 windows."""
    total = _align_frame_count(total_frames)
    window = _align_frame_count(window_frames)
    context = int(context_frames)
    if context < 5 or context % 17 != 5:
        raise ValueError("context_frames must be an H3 5 + 17*n run")
    if context >= window:
        raise ValueError("context_frames must be shorter than window_frames")
    stride = window - context
    starts = [0]
    while starts[-1] + window < total:
        starts.append(starts[-1] + stride)
    plan = [
        (start, min(window, _align_frame_count(total - start)))
        for start in starts
    ]
    if any(i and length <= context for i, (_, length) in enumerate(plan)):
        raise ValueError("a continuation window cannot contain only context")
    return plan


def _av_parts(latent: Mapping) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(latent, Mapping) or "samples" not in latent:
        raise ValueError("expected a MiniMax H3 AV LATENT")
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = tuple(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = tuple(samples)
    else:
        raise ValueError(f"unsupported H3 nested latent: {type(samples)!r}")
    if len(parts) != 2:
        raise ValueError(f"expected video and audio streams, got {len(parts)}")
    video, audio = parts
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError(
            f"unexpected H3 AV layout: video={tuple(video.shape)}, "
            f"audio={tuple(audio.shape)}"
        )
    if video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError("MiniMax H3 music-video generation supports batch size 1")
    if video.shape[1] != 24 or audio.shape[1:3] != (32, 2):
        raise ValueError(
            "expected H3 channels video=24 and audio=[32,2], got "
            f"video={video.shape[1]}, audio={tuple(audio.shape[1:3])}"
        )
    return video, audio


def _validate_audio(audio: Mapping) -> tuple[torch.Tensor, int]:
    if not isinstance(audio, Mapping):
        raise ValueError("master_audio must be a connected ComfyUI AUDIO value")
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if waveform is None or sample_rate is None or not isinstance(
        waveform, torch.Tensor
    ):
        raise ValueError("master_audio is missing waveform or sample_rate")
    if waveform.ndim != 3 or min(waveform.shape) < 1:
        raise ValueError("master_audio waveform must be [B,C,samples]")
    if waveform.shape[0] != 1:
        raise ValueError("master_audio must have batch size 1")
    if waveform.shape[1] not in (1, 2):
        raise ValueError("master_audio must be mono or stereo")
    return waveform, int(sample_rate)


def _prepare_master_waveform(audio_vae, master_audio: Mapping) -> torch.Tensor:
    """Return stereo master audio at the Audio-VAE sample rate."""
    waveform, sample_rate = _validate_audio(master_audio)
    waveform = waveform[:1]
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sample_rate != vae_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, vae_rate)
    return waveform


def _fit_waveform(waveform: torch.Tensor, wanted: int) -> torch.Tensor:
    wanted = int(wanted)
    if waveform.shape[-1] >= wanted:
        return waveform[..., :wanted]
    return torch.nn.functional.pad(waveform, (0, wanted - waveform.shape[-1]))


def _encode_master_window(audio_vae, master_waveform: torch.Tensor,
                          start_frame: int, frame_count: int,
                          target_audio: torch.Tensor) -> torch.Tensor:
    """Encode one absolute master-audio interval onto the target audio grid."""
    sample_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    start_sample = _sample_at_frame(start_frame, sample_rate)
    picture_end = _sample_at_frame(start_frame + frame_count, sample_rate)
    picture_samples = picture_end - start_sample
    expected_steps = int(target_audio.shape[-1])
    grid_samples = math.ceil(expected_steps / H3_AUDIO_LATENT_FPS * sample_rate)
    encode_samples = max(picture_samples, grid_samples)
    chunk = _fit_waveform(
        master_waveform[..., start_sample:start_sample + encode_samples],
        encode_samples,
    )
    encoded = audio_vae.encode(chunk.movedim(1, -1))
    if encoded.ndim != 4 or encoded.shape[1:3] != target_audio.shape[1:3]:
        raise ValueError(
            "H3 Audio VAE returned an incompatible latent: "
            f"encoded={tuple(encoded.shape)}, target={tuple(target_audio.shape)}"
        )
    if encoded.shape[-1] < expected_steps:
        missing = expected_steps - int(encoded.shape[-1])
        retry_samples = encode_samples + math.ceil(
            (missing + 1) * sample_rate / H3_AUDIO_LATENT_FPS
        )
        chunk = _fit_waveform(
            master_waveform[..., start_sample:start_sample + retry_samples],
            retry_samples,
        )
        encoded = audio_vae.encode(chunk.movedim(1, -1))
    if encoded.shape[-1] < expected_steps:
        raise RuntimeError(
            f"Audio VAE produced {encoded.shape[-1]}/{expected_steps} latent steps"
        )
    return encoded[:1, ..., :expected_steps].to(
        device=target_audio.device, dtype=target_audio.dtype
    )


def _prepare_masked_target(latent: Mapping, audio_vae,
                           master_waveform: torch.Tensor,
                           start_frame: int, frame_count: int,
                           context_frames: int,
                           source_latent: Mapping | None) -> dict:
    """Write fixed audio and an optional protected video prefix into target."""
    target_video, target_audio = _av_parts(latent)
    if _pixel_frames(target_video.shape[2]) != int(frame_count):
        raise RuntimeError("official H3 target does not match the planned frame run")
    out_video = target_video.clone()
    out_audio = _encode_master_window(
        audio_vae, master_waveform, start_frame, frame_count, target_audio
    )

    video_mask = torch.ones(
        (1, 1, out_video.shape[2], out_video.shape[3], out_video.shape[4]),
        device=out_video.device,
        dtype=torch.float32,
    )
    if source_latent is not None:
        source_video, _ = _av_parts(source_latent)
        context_steps = _video_latent_steps(context_frames)
        if context_steps >= out_video.shape[2] or context_steps > source_video.shape[2]:
            raise ValueError("video context is too long for source or target window")
        tail_start = int(source_video.shape[2]) - context_steps
        if tail_start % len(FRAME_PER_TOKEN):
            raise RuntimeError(
                f"source video tail is off the H3 temporal phase: {tail_start}"
            )
        if tuple(source_video.shape[1:2] + source_video.shape[3:]) != tuple(
            out_video.shape[1:2] + out_video.shape[3:]
        ):
            raise ValueError("source and target video latent geometry differs")
        out_video[:, :, :context_steps] = source_video[
            :, :, -context_steps:
        ].to(device=out_video.device, dtype=out_video.dtype)
        video_mask[:, :, :context_steps] = 0.0

    audio_mask = torch.zeros(
        (1, 1, out_audio.shape[2], out_audio.shape[3]),
        device=out_audio.device,
        dtype=torch.float32,
    )
    prepared = dict(latent)
    prepared["samples"] = comfy.nested_tensor.NestedTensor((out_video, out_audio))
    prepared["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (video_mask, audio_mask)
    )
    return prepared


def _make_native_av_sampling(model, video_shift: float, audio_shift: float):
    sampling_cls = getattr(comfy.model_sampling, "ModelSamplingAV", None)
    if sampling_cls is None:
        raise RuntimeError("current ComfyUI ModelSamplingAV support is required")

    class H3NativeAVSampling(sampling_cls, comfy.model_sampling.CONST):
        pass

    sampling = H3NativeAVSampling(model.model.model_config)
    sampling.set_parameters(shift=video_shift, audio_shift=audio_shift)
    original = model.get_model_object("model_sampling")
    if hasattr(original, "noise_scale"):
        sampling.set_noise_scale(original.noise_scale)
    return sampling


def _patch_h3_model(model, sampling, video_shift: float, audio_shift: float):
    patched = model.clone()
    patched.add_object_patch("model_sampling", sampling)
    options = patched.model_options.get("transformer_options", {}).copy()
    options["minimax_h3_sigma_shift_video"] = video_shift
    options["minimax_h3_sigma_shift_audio"] = audio_shift
    patched.model_options["transformer_options"] = options
    return patched


class _BasicPositiveGuider(comfy.samplers.CFGGuider):
    def set_positive(self, positive):
        self.inner_set_conds({"positive": positive})


class MaskedAVWindowSampler:
    """Run stock H3 sampling with nested per-stream denoise masks."""

    def sample(self, model, positive, latent, seed, steps, sampler_name,
               scheduler, shift_video, shift_audio):
        sampling = _make_native_av_sampling(model, shift_video, shift_audio)
        sampling_model = _patch_h3_model(
            model, sampling, shift_video, shift_audio
        )
        sigmas = comfy.samplers.calculate_sigmas(
            sampling, str(scheduler), int(steps)
        ).cpu()
        if len(sigmas) < 2 or not torch.isclose(
            sigmas[-1], sigmas.new_tensor(0.0)
        ):
            raise ValueError("scheduler must end at sigma zero")
        samples = latent["samples"]
        noise = comfy.sample.prepare_noise(
            samples, int(seed), latent.get("batch_index")
        )
        guider = _BasicPositiveGuider(sampling_model)
        guider.set_positive(positive)
        sampled = guider.sample(
            noise,
            samples,
            comfy.samplers.sampler_object(str(sampler_name)),
            sigmas,
            denoise_mask=latent.get("noise_mask"),
            callback=None,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            seed=int(seed),
        )
        out = dict(latent)
        out.pop("noise_mask", None)
        out["samples"] = sampled
        return out


class PixelCrossfadeAssembler:
    """Collect decoded clips while retaining only the next seam tail."""

    def __init__(self, overlap_frames: int):
        self.overlap = max(0, int(overlap_frames))
        self.parts: list[torch.Tensor] = []
        self.tail: torch.Tensor | None = None
        self.started = False

    def add(self, decoded: torch.Tensor, context_frames: int) -> None:
        if decoded.ndim != 4 or decoded.shape[0] < 1:
            raise ValueError("Video VAE decode must return IMAGE [T,H,W,C]")
        decoded = decoded.detach().to(device="cpu", dtype=torch.float32)
        context = max(0, int(context_frames))
        overlap = min(self.overlap, context)
        if not self.started:
            self.started = True
            if overlap:
                self.parts.append(decoded[:-overlap])
                self.tail = decoded[-overlap:]
            else:
                self.parts.append(decoded)
            return

        if context > decoded.shape[0]:
            raise ValueError("decoded continuation is shorter than its context")
        if overlap:
            if self.tail is None or self.tail.shape[0] != overlap:
                raise RuntimeError("retained crossfade tail length mismatch")
            dst = decoded[context - overlap:context]
            alpha = torch.linspace(0.0, 1.0, overlap + 2)[1:-1].view(
                -1, 1, 1, 1
            )
            seam = self.tail * (1.0 - alpha) + dst * alpha
            fresh = torch.cat((seam, decoded[context:]), dim=0)
            self.parts.append(fresh[:-overlap])
            self.tail = fresh[-overlap:]
        else:
            self.parts.append(decoded[context:])

    def finish(self) -> torch.Tensor:
        if self.tail is not None:
            self.parts.append(self.tail)
            self.tail = None
        if not self.parts:
            raise RuntimeError("no decoded video clips were assembled")
        return torch.cat(self.parts, dim=0)


def _slice_audio_window(audio: Mapping, start_frame: int,
                        frame_count: int) -> dict:
    waveform, sample_rate = _validate_audio(audio)
    start = _sample_at_frame(start_frame, sample_rate)
    end = _sample_at_frame(start_frame + frame_count, sample_rate)
    chunk = _fit_waveform(waveform[..., start:end], end - start)
    return {"waveform": chunk, "sample_rate": sample_rate}


def _slice_video_window(video: torch.Tensor, start_frame: int,
                        frame_count: int) -> torch.Tensor:
    if not isinstance(video, torch.Tensor) or video.ndim != 4 or video.shape[0] < 1:
        raise ValueError("reference video must be IMAGE [T,H,W,C]")
    start = max(0, int(start_frame))
    wanted = max(1, int(frame_count))
    chunk = video[start:start + wanted]
    if chunk.shape[0] == 0:
        chunk = video[-1:].clone()
    if chunk.shape[0] < wanted:
        chunk = torch.cat((
            chunk,
            chunk[-1:].expand(wanted - chunk.shape[0], *chunk.shape[1:]),
        ), dim=0)
    return chunk


def _unpack_official_conditioning(result):
    values = result.args if hasattr(result, "args") else result
    if not isinstance(values, (tuple, list)) or len(values) < 2:
        raise RuntimeError("MiniMax H3 Reference to Video must return two outputs")
    return values[0], values[1]
