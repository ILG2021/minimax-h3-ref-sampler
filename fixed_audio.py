"""All-in-one ComfyUI sampler for MiniMax H3 Ref2VA with fixed target audio."""
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
from comfy.k_diffusion.sampling import to_d

H3_VIDEO_FPS = 24
H3_AUDIO_LATENT_FPS = 40


def _video_latent_steps(frame_count: int) -> int:
    """Native H3 temporal VAE size for a supported 5 + 17*n frame run."""
    if frame_count < 5 or frame_count % 17 != 5:
        raise ValueError("H3 frame count must be 5 + 17*n")
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def _context_latent_steps(frame_count: int) -> int:
    """Latent steps occupied by one exact H3 continuation span."""
    return _video_latent_steps(frame_count)


def _align_frame_count(frame_count: int) -> int:
    frame_count = max(5, int(frame_count))
    return frame_count + ((5 - frame_count) % 17)


def _plan_windows(total_frames: int, window_frames: int,
                  context_frames: int) -> list[tuple[int, int]]:
    """Return absolute `(start, sample_length)` windows on the H3 frame grid."""
    total = _align_frame_count(total_frames)
    window = _align_frame_count(window_frames)
    context = int(context_frames)
    if context < 5 or context % 17 != 5:
        raise ValueError("context_frames must be one of H3's 5 + 17*n spans")
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
    if any(index > 0 and length <= context
           for index, (_, length) in enumerate(plan)):
        raise ValueError("a continuation window cannot contain only context")
    return plan


def _av_parts(latent: Mapping) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and split H3 NestedTensor(video, audio)."""
    if not isinstance(latent, Mapping) or "samples" not in latent:
        raise ValueError("latent must be a MiniMax H3 AV LATENT")
    samples = latent["samples"]
    if not getattr(samples, "is_nested", False):
        raise ValueError("latent.samples must be NestedTensor(video, audio)")
    parts = tuple(samples.unbind())
    if len(parts) != 2:
        raise ValueError(f"expected two H3 latent parts, got {len(parts)}")
    video, audio = parts
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError(
            "unexpected H3 AV layout: "
            f"video={tuple(video.shape)}, audio={tuple(audio.shape)}"
        )
    if video.shape[0] != audio.shape[0]:
        raise ValueError("video/audio latent batch sizes differ")
    if video.shape[1] != 24 or audio.shape[1:3] != (32, 2):
        raise ValueError(
            "expected H3 channels video=24 and audio=[32,2], got "
            f"video={video.shape[1]}, audio={tuple(audio.shape[1:3])}"
        )
    return video, audio


def _validate_audio(audio: Mapping) -> tuple[torch.Tensor, int]:
    if not isinstance(audio, Mapping):
        raise ValueError("target_audio must be a connected ComfyUI AUDIO value")
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor) or sample_rate is None:
        raise ValueError("target_audio is missing waveform or sample_rate")
    if waveform.ndim != 3 or min(waveform.shape) < 1:
        raise ValueError(
            "target_audio.waveform must be non-empty [batch, channels, samples]"
        )
    if waveform.shape[0] != 1:
        raise ValueError("MiniMax H3 fixed-audio sampling supports batch size 1")
    return waveform, int(sample_rate)


def _fit_audio_latent(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Crop or zero-pad encoded source audio to the target H3 duration."""
    if source.ndim != 4:
        raise ValueError(
            f"H3 Audio VAE must return [B,32,2,T], got {tuple(source.shape)}"
        )
    if source.shape[1:3] != target.shape[1:3]:
        raise ValueError(
            "Audio VAE latent layout mismatch: "
            f"source={tuple(source.shape)}, target={tuple(target.shape)}"
        )
    if source.shape[0] != target.shape[0]:
        if source.shape[0] == 1:
            source = source.expand(target.shape[0], -1, -1, -1)
        else:
            raise ValueError("audio latent batch cannot match target AV latent")
    target_t = target.shape[-1]
    source = source[..., :target_t]
    if source.shape[-1] < target_t:
        source = torch.nn.functional.pad(source, (0, target_t - source.shape[-1]))
    return source.to(device=target.device, dtype=target.dtype)


def _shift_sigma(base_sigma, shift: float):
    return shift * base_sigma / (1.0 + (shift - 1.0) * base_sigma)


def _audio_sigma(video_sigma, video_shift: float, audio_shift: float):
    """Invert the video shift and apply the audio shift on the shared flow time."""
    base_sigma = video_sigma / (
        video_shift + video_sigma * (1.0 - video_shift)
    )
    return _shift_sigma(base_sigma, audio_shift)


def _make_model_sampling(model, video_shift: float):
    class H3ExplicitAudioClockSampling(
        comfy.model_sampling.ModelSamplingDiscreteFlow,
        comfy.model_sampling.CONST,
    ):
        @property
        def audio_scale(self):
            # The sampler presents audio on its real audio clock. Returning 1
            # prevents native ModelSamplingAV from carrying/scaling it again.
            return 1.0

    sampling = H3ExplicitAudioClockSampling(model.model.model_config)
    sampling.set_parameters(shift=video_shift)
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


def _make_fixed_audio_sampler(
    clean_audio: torch.Tensor,
    audio_noise: torch.Tensor,
    video_noise: torch.Tensor | None,
    video_values: int,
    video_shift: float,
    audio_shift: float,
    context_mask_p: torch.Tensor | None,
    context_latent_p: torch.Tensor | None,
):
    """Build Euler sampler that integrates video only and reconstructs audio."""
    packed_values = video_values + math.prod(clean_audio.shape[1:])
    clean_cpu = clean_audio.detach().to(device="cpu", dtype=torch.float32)
    noise_cpu = audio_noise.detach().to(device="cpu", dtype=torch.float32)
    video_noise_cpu = None
    if video_noise is not None:
        video_noise_cpu = video_noise.detach().to(
            device="cpu", dtype=torch.float32
        )

    def sample_fixed_audio(model_wrap, x, sigmas, extra_args=None, callback=None,
                           disable=None):
        # ComfyUI's pack_latents() contract is [B, 1, packed_values]. Keep
        # this singleton axis: unpack_latents() requires it after sampling.
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[-1] != packed_values:
            raise ValueError(
                f"packed H3 latent mismatch: expected [B,1,{packed_values}], "
                f"got {tuple(x.shape)}"
            )
        args = {} if extra_args is None else extra_args
        clean = clean_cpu.to(device=x.device, dtype=x.dtype).reshape(x.shape[0], 1, -1)
        noise = noise_cpu.to(device=x.device, dtype=x.dtype).reshape(x.shape[0], 1, -1)

        sigma_batch = x.new_ones([x.shape[0]])
        video = x[..., :video_values]  # pure noise video

        context_mask = None
        context_latent = None
        target_noise = None
        if context_mask_p is not None and context_latent_p is not None:
            context_mask = context_mask_p.to(
                device=x.device, dtype=x.dtype
            ).reshape(x.shape[0], 1, -1)
            context_latent = context_latent_p.to(
                device=x.device, dtype=x.dtype
            ).reshape(x.shape[0], 1, -1)
            if video_noise_cpu is None:
                raise ValueError("video noise is required for motion context")
            target_noise = video_noise_cpu.to(
                device=x.device, dtype=x.dtype
            ).reshape(x.shape[0], 1, -1)
            if context_mask.shape[-1] != video_values:
                raise ValueError(
                    "context mask packed size mismatch: "
                    f"expected {video_values}, got {context_mask.shape[-1]}"
                )
            target_noisy = (
                (1.0 - sigmas[0]) * context_latent
                + sigmas[0] * target_noise
            )
            video = video * context_mask + target_noisy * (1.0 - context_mask)

        for step in comfy.utils.model_trange(len(sigmas) - 1, disable=disable):
            sigma_v = sigmas[step]
            sigma_v_next = sigmas[step + 1]
            sigma_a = _audio_sigma(sigma_v, video_shift, audio_shift)
            sigma_a_next = _audio_sigma(sigma_v_next, video_shift, audio_shift)

            # Rectified-flow forward path with one fixed epsilon_a.
            audio_t = (1.0 - sigma_a) * clean + sigma_a * noise
            current = torch.cat((video, audio_t), dim=-1)

            # model_wrap already carries BasicGuider's positive/minimax_refs.
            denoised = model_wrap(current, sigma_v * sigma_batch, **args)
            derivative = to_d(current, sigma_v, denoised)

            # Integrate only target video on the outer video clock.
            video = video + derivative[..., :video_values] * (sigma_v_next - sigma_v)
            if context_mask is not None and context_latent is not None:
                target_noisy = (
                    (1.0 - sigma_v_next) * context_latent
                    + sigma_v_next * target_noise
                )
                video = (
                    video * context_mask
                    + (1.0 - context_mask) * target_noisy
                )
            # The model's audio prediction is deliberately ignored. Build the
            # exact next audio state from a0, epsilon_a and sigma_audio(next).
            audio_next = (1.0 - sigma_a_next) * clean + sigma_a_next * noise
            current = torch.cat((video, audio_next), dim=-1)

            if callback is not None:
                callback({
                    "x": current,
                    "i": step,
                    "sigma": sigma_v,
                    "sigma_hat": sigma_v,
                    # Preview must expose the known clean audio endpoint.
                    "denoised": torch.cat(
                        (denoised[..., :video_values], clean), dim=-1
                    ),
                })

        # Last audio sigma is expected to be zero, but force a0 explicitly.
        return torch.cat((video, clean), dim=-1)

    sample_fixed_audio.__name__ = "sample_minimax_h3_ref_fixed_audio_euler"
    return comfy.samplers.KSAMPLER(sample_fixed_audio)


def _make_video_context_sampler(video_values, context_mask_p,
                                context_latent_p, video_noise_p):
    """Build Euler sampling that pins only the overlapping video prefix."""
    mask_cpu = context_mask_p.detach().to(device="cpu", dtype=torch.float32)
    target_cpu = context_latent_p.detach().to(device="cpu", dtype=torch.float32)
    noise_cpu = video_noise_p.detach().to(device="cpu", dtype=torch.float32)

    def sample_context(model_wrap, x, sigmas, extra_args=None, callback=None,
                       disable=None):
        args = {} if extra_args is None else extra_args
        mask = mask_cpu.to(x.device, x.dtype).reshape(x.shape[0], 1, -1)
        target = target_cpu.to(x.device, x.dtype).reshape(x.shape[0], 1, -1)
        noise = noise_cpu.to(x.device, x.dtype).reshape(x.shape[0], 1, -1)
        if not (mask.shape[-1] == target.shape[-1] == noise.shape[-1]
                == video_values):
            raise ValueError("motion-context tensors do not match packed video size")
        sigma_batch = x.new_ones([x.shape[0]])
        target_noisy = (1.0 - sigmas[0]) * target + sigmas[0] * noise
        x = torch.cat((
            x[..., :video_values] * mask + target_noisy * (1.0 - mask),
            x[..., video_values:],
        ), dim=-1)
        for step in comfy.utils.model_trange(len(sigmas) - 1, disable=disable):
            sigma = sigmas[step]
            sigma_next = sigmas[step + 1]
            denoised = model_wrap(x, sigma * sigma_batch, **args)
            x = x + to_d(x, sigma, denoised) * (sigma_next - sigma)
            target_noisy = (1.0 - sigma_next) * target + sigma_next * noise
            video = (
                x[..., :video_values] * mask
                + target_noisy * (1.0 - mask)
            )
            x = torch.cat((video, x[..., video_values:]), dim=-1)
            if callback is not None:
                callback({"x": x, "i": step, "sigma": sigma,
                          "sigma_hat": sigma, "denoised": denoised})
        return x

    sample_context.__name__ = "sample_minimax_h3_video_context_euler"
    return comfy.samplers.KSAMPLER(sample_context)


class _BasicPositiveGuider(comfy.samplers.CFGGuider):
    """Same one-condition setup used by ComfyUI's Basic Guider node."""

    def set_positive(self, positive):
        self.inner_set_conds({"positive": positive})


class _FixedAudioWindowSampler:
    """Sample one H3 window with fixed audio and optional video-tail context."""

    def sample(self, model, positive, latent, clean_audio, seed, steps,
               scheduler, shift_video, shift_audio,
               audio_noise=None, context_video_latent=None, context_frames=0):
        video, audio_template = _av_parts(latent)
        if video.shape[0] != 1:
            raise ValueError("MiniMax H3 Ref Sampler supports batch size 1")
        fixed_audio = clean_audio is not None
        if fixed_audio:
            clean_audio = _fit_audio_latent(clean_audio, audio_template)
        context_latent = None
        context_mask = None

        if context_video_latent is not None:
            context = context_video_latent
            if not isinstance(context, torch.Tensor) or context.ndim != 5:
                raise ValueError("continuation context must be [B,24,T,H,W]")
            context_t = _context_latent_steps(int(context_frames))
            if context_t >= video.shape[2]:
                raise ValueError("continuation context must be shorter than the window")
            expected = (
                video.shape[0], video.shape[1], context_t,
                video.shape[3], video.shape[4],
            )
            if tuple(context.shape) != tuple(expected):
                raise ValueError(
                    "continuation context layout mismatch: "
                    f"expected {expected}, got {tuple(context.shape)}"
                )
            context_latent = torch.zeros_like(video)
            context_latent[:, :, :context_t] = context.to(
                device=video.device, dtype=video.dtype
            )
            context_mask = torch.ones_like(video)
            context_mask[:, :, :context_t] = 0.0

        # Do not pass an inherited inpaint mask into model_wrap. Its generic
        # video-clock blending could overwrite sigma-consistent target audio.
        prepared = dict(latent)
        prepared.pop("noise_mask", None)
        prepared["samples"] = comfy.nested_tensor.NestedTensor(
            (video, clean_audio if fixed_audio else audio_template)
        )

        if fixed_audio:
            sampling = _make_model_sampling(model, shift_video)
            sampling_model = _patch_h3_model(
                model, sampling, shift_video, shift_audio
            )
        else:
            sampling = model.get_model_object("model_sampling")
            sampling_model = model
        sigmas = comfy.samplers.calculate_sigmas(
            sampling, scheduler, int(steps)
        ).cpu()
        # Some discrete schedulers legitimately collapse duplicate timesteps,
        # so the resulting list may be shorter than steps+1.
        if len(sigmas) < 2 or not torch.isclose(
            sigmas[-1], sigmas.new_tensor(0.0)
        ):
            raise ValueError("scheduler must return at least two sigmas ending at zero")

        latent_samples = prepared["samples"]
        noise = comfy.sample.prepare_noise(
            latent_samples, int(seed), prepared.get("batch_index")
        )
        video_noise = None
        if context_mask is not None:
            if not getattr(noise, "is_nested", False):
                raise ValueError("expected nested H3 noise for motion context")
            noise_parts = tuple(noise.unbind())
            if len(noise_parts) != 2:
                raise ValueError("expected video/audio H3 noise components")
            video_noise = noise_parts[0] * float(
                getattr(sampling, "noise_scale", 1.0)
            )
        if fixed_audio:
            if audio_noise is None or tuple(audio_noise.shape) != tuple(clean_audio.shape):
                raise ValueError("fixed audio noise does not match the encoded window")
            audio_noise = audio_noise * float(getattr(sampling, "noise_scale", 1.0))
            sampler = _make_fixed_audio_sampler(
                clean_audio, audio_noise, video_noise,
                math.prod(video.shape[1:]), shift_video, shift_audio,
                context_mask, context_latent,
            )
        elif context_mask is not None:
            sampler = _make_video_context_sampler(
                math.prod(video.shape[1:]), context_mask, context_latent,
                video_noise,
            )
        else:
            sampler = comfy.samplers.sampler_object("euler")

        guider = _BasicPositiveGuider(sampling_model)
        guider.set_positive(positive)
        samples = guider.sample(
            noise,
            latent_samples,
            sampler,
            sigmas,
            denoise_mask=None,
            callback=None,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            seed=int(seed),
        )

        output = prepared.copy()
        output["samples"] = samples
        return output


def _slice_audio_window(audio: Mapping, start_frame: int, frame_count: int) -> dict:
    """Cut a mono reference-audio window and zero-pad its tail."""
    waveform, sample_rate = _validate_audio(audio)
    if waveform.shape[1] != 1:
        waveform = waveform.mean(dim=1, keepdim=True)
    start = round(int(start_frame) / H3_VIDEO_FPS * sample_rate)
    wanted = max(1, round(int(frame_count) / H3_VIDEO_FPS * sample_rate))
    chunk = waveform[..., start:start + wanted]
    if chunk.shape[-1] < wanted:
        chunk = torch.nn.functional.pad(chunk, (0, wanted - chunk.shape[-1]))
    return {"waveform": chunk, "sample_rate": sample_rate}


def _slice_video_window(video: torch.Tensor, start_frame: int,
                        frame_count: int) -> torch.Tensor:
    """Cut one IMAGE timeline window and hold its final frame when short."""
    if not isinstance(video, torch.Tensor) or video.ndim != 4 or video.shape[0] < 1:
        raise ValueError("reference video must be a non-empty IMAGE batch [T,H,W,C]")
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
        raise RuntimeError(
            f"MiniMax H3 Reference to Video returned {type(result)!r}, expected two outputs"
        )
    return values[0], values[1]
