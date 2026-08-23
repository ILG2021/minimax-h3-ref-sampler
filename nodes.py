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


def _video_frames_from_latent_steps(latent_t: int) -> int:
    if latent_t < 2 or (latent_t - 2) % 5:
        raise ValueError(f"unexpected H3 video latent length: {latent_t}")
    return 5 + ((latent_t - 2) // 5) * 17


def _align_frame_count(frame_count: int) -> int:
    frame_count = max(5, int(frame_count))
    return frame_count + ((5 - frame_count) % 17)


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
    video_mask_p: torch.Tensor | None,
    target_latent_p: torch.Tensor | None,
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

        video_mask = None
        target_latent = None
        target_noise = None
        if video_mask_p is not None and target_latent_p is not None:
            video_mask = video_mask_p.to(
                device=x.device, dtype=x.dtype
            ).reshape(x.shape[0], 1, -1)
            target_latent = target_latent_p.to(
                device=x.device, dtype=x.dtype
            ).reshape(x.shape[0], 1, -1)
            if video_noise_cpu is None:
                raise ValueError("video noise is required for video repaint")
            target_noise = video_noise_cpu.to(
                device=x.device, dtype=x.dtype
            ).reshape(x.shape[0], 1, -1)
            if video_mask.shape[-1] != video_values:
                raise ValueError(
                    "video mask packed size mismatch: "
                    f"expected {video_values}, got {video_mask.shape[-1]}"
                )
            target_noisy = (
                (1.0 - sigmas[0]) * target_latent
                + sigmas[0] * target_noise
            )
            video = video * video_mask + target_noisy * (1.0 - video_mask)

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
            if video_mask is not None and target_latent is not None:
                target_noisy = (
                    (1.0 - sigma_v_next) * target_latent
                    + sigma_v_next * target_noise
                )
                video = (
                    video * video_mask
                    + (1.0 - video_mask) * target_noisy
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


def _make_video_mask_only_sampler(
    video_values: int,
    video_mask: torch.Tensor,
    target_latent: torch.Tensor,
    video_noise: torch.Tensor,
):
    """Build a standard Euler sampler with a sigma-consistent video mask."""
    mask_cpu = video_mask.detach().to(device="cpu", dtype=torch.float32)
    target_cpu = target_latent.detach().to(device="cpu", dtype=torch.float32)
    noise_cpu = video_noise.detach().to(device="cpu", dtype=torch.float32)

    def sample_masked(model_wrap, x, sigmas, extra_args=None, callback=None,
                      disable=None):
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[-1] <= video_values:
            raise ValueError(
                "expected packed H3 AV latent [B,1,video+audio], got "
                f"{tuple(x.shape)}"
            )
        args = {} if extra_args is None else extra_args
        mask = mask_cpu.to(x.device, x.dtype).reshape(x.shape[0], 1, -1)
        target = target_cpu.to(x.device, x.dtype).reshape(x.shape[0], 1, -1)
        noise = noise_cpu.to(x.device, x.dtype).reshape(x.shape[0], 1, -1)
        if not (mask.shape[-1] == target.shape[-1] == noise.shape[-1]
                == video_values):
            raise ValueError("masked video tensors do not match packed video size")
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

            target_noisy = (
                (1.0 - sigma_next) * target + sigma_next * noise
            )
            video = (
                x[..., :video_values] * mask
                + target_noisy * (1.0 - mask)
            )
            x = torch.cat((video, x[..., video_values:]), dim=-1)
            if callback is not None:
                callback({
                    "x": x,
                    "i": step,
                    "sigma": sigma,
                    "sigma_hat": sigma,
                    "denoised": denoised,
                })
        return x

    sample_masked.__name__ = "sample_minimax_h3_masked_euler"
    return comfy.samplers.KSAMPLER(sample_masked)


class _BasicPositiveGuider(comfy.samplers.CFGGuider):
    """Same one-condition setup used by ComfyUI's Basic Guider node."""

    def set_positive(self, positive):
        self.inner_set_conds({"positive": positive})


class MinimaxH3RefSampler:
    """Replace the standard H3 sampling stack and preserve target audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                }),
                "steps": ("INT", {"default": 8, "min": 1, "max": 1000}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "beta"}),
                "shift_video": ("FLOAT", {
                    "default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01,
                }),
                "shift_audio": ("FLOAT", {
                    "default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01,
                }),
            },
            "optional":{
                "audio_vae": ("VAE",),
                "target_audio": ("AUDIO",),
                "context_frames": (["5", "22", "39", "56"], {"default": "22"}),
                "video_vae": ("VAE",),
                "video_mask": ("MASK",),
                "target_video": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LATENT", "AUDIO")
    RETURN_NAMES = ("samples", "original_audio")
    FUNCTION = "sample"
    CATEGORY = "sampling/minimax"

    def sample(self, model, positive, latent, seed, steps, scheduler,
               shift_video, shift_audio, audio_vae=None, target_audio=None,
               context_frames="22", video_vae=None, video_mask=None,
               target_video=None, _context_video_latent=None,
               _context_frames=0, _single_window=False):
        video, audio_template = _av_parts(latent) # video: [B, 24, T, H, W] audio: [B, 32, 2, T]
        if video.shape[0] != 1:
            raise ValueError("MiniMax H3 Ref Sampler supports batch size 1")

        if (audio_vae is None) != (target_audio is None):
            raise ValueError(
                "audio_vae and target_audio must either both be connected or both omitted"
            )
        if target_audio is not None and not _single_window:
            waveform, sample_rate = _validate_audio(target_audio)
            total_audio_t = round(
                waveform.shape[-1] / sample_rate * H3_AUDIO_LATENT_FPS
            )
            if total_audio_t > audio_template.shape[-1]:
                return self._sample_long(
                    model, positive, latent, audio_vae, target_audio, seed,
                    steps, scheduler, shift_video, shift_audio,
                    context_frames, video_vae, video_mask, target_video,
                )
        clean_audio = None
        original_audio = None
        if target_audio is not None:
            waveform, sample_rate = _validate_audio(target_audio)
            vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
            if sample_rate != vae_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, vae_rate
                )
            clean_audio = audio_vae.encode(waveform.movedim(1, -1))
            clean_audio = _fit_audio_latent(clean_audio, audio_template)
            # The latent follows the window duration, but the public audio
            # output is a true bypass: no resample, crop, pad, or VAE round trip.
            original_audio = dict(target_audio)
        target_latent = None
        if (video_mask is None) != (target_video is None):
            raise ValueError(
                "video_mask and target_video must either both be connected or both omitted"
            )
        if video_mask is not None:
            if video_vae is None:
                raise ValueError("video_vae is required when using video repaint")
            # H3 FaceRefine conversion: IMAGE batches are video frames
            # [T,H,W,C]. Encode RGB, then convert [T,C,H,W] to H3's
            # [B,C,T,H,W] layout when the VAE returns a 4D tensor.
            target_latent = video_vae.encode(target_video[..., :3])
            if target_latent.ndim == 4:
                target_latent = target_latent.unsqueeze(0).movedim(1, 2)
            if target_latent.ndim != 5:
                raise ValueError(
                    "Video VAE must return [B,C,T,H,W] (or [T,C,H,W]), got "
                    f"{tuple(target_latent.shape)}"
                )
            if tuple(target_latent.shape) != tuple(video.shape):
                raise ValueError(
                    "target video latent must match the H3 target layout: "
                    f"target={tuple(target_latent.shape)}, H3={tuple(video.shape)}"
                )

            # MASK is a frame batch [T,H,W]. Keep it independent from the
            # VAE, resize it to latent T/H/W, then expand it across all H3
            # video channels so its packed order matches the video latent.
            video_mask = video_mask.float()
            if video_mask.ndim == 2:
                video_mask = video_mask.unsqueeze(0)
            if video_mask.ndim == 3:
                video_mask = video_mask.unsqueeze(0).unsqueeze(0)
            elif video_mask.ndim == 4:
                video_mask = video_mask.unsqueeze(1)
            else:
                raise ValueError(
                    "video_mask must be [T,H,W] or [B,T,H,W], got "
                    f"{tuple(video_mask.shape)}"
                )
            if video_mask.shape[0] != video.shape[0]:
                raise ValueError("video mask batch size differs from H3 video batch")
            video_mask = torch.nn.functional.interpolate(
                video_mask,
                size=video.shape[2:],
                mode="trilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            video_mask = video_mask.expand(
                video.shape[0], video.shape[1], *video.shape[2:]
            )

        # Private continuation path. It is deliberately separate from the
        # public repaint sockets: continuation always pins a full head run.
        if _context_video_latent is not None:
            if video_mask is not None:
                raise ValueError("continuation context cannot be combined with repaint")
            context = _context_video_latent
            if not isinstance(context, torch.Tensor) or context.ndim != 5:
                raise ValueError("continuation context must be [B,24,T,H,W]")
            context_t = _context_latent_steps(int(_context_frames))
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
            target_latent = torch.zeros_like(video)
            target_latent[:, :, :context_t] = context.to(
                device=video.device, dtype=video.dtype
            )
            video_mask = torch.ones_like(video)
            video_mask[:, :, :context_t] = 0.0

        # Do not pass an inherited inpaint mask into model_wrap. Its generic
        # video-clock blending could overwrite sigma-consistent target audio.
        prepared = dict(latent)
        prepared.pop("noise_mask", None)
        prepared["samples"] = comfy.nested_tensor.NestedTensor(
            (video, clean_audio if clean_audio is not None else audio_template)
        )

        fixed_audio = clean_audio is not None
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
        if video_mask is not None:
            if not getattr(noise, "is_nested", False):
                raise ValueError("expected nested H3 noise for video repaint")
            noise_parts = tuple(noise.unbind())
            if len(noise_parts) != 2:
                raise ValueError("expected video/audio H3 noise components")
            video_noise = noise_parts[0] * float(
                getattr(sampling, "noise_scale", 1.0)
            )
        if fixed_audio:
            # Dedicated fixed audio epsilon_a, generated once and reused.
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            audio_noise = torch.randn(
                clean_audio.shape, generator=generator, dtype=torch.float32
            )
            audio_noise *= float(getattr(sampling, "noise_scale", 1.0))
            sampler = _make_fixed_audio_sampler(
                clean_audio,
                audio_noise,
                video_noise,
                math.prod(video.shape[1:]),
                shift_video,
                shift_audio,
                video_mask,
                target_latent,
            )
        elif video_mask is not None:
            sampler = _make_video_mask_only_sampler(
                math.prod(video.shape[1:]),
                video_mask,
                target_latent,
                video_noise,
            )
        else:
            # No optional feature is active: use ComfyUI's stock Euler path.
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
        return output, original_audio


    def _sample_long(self, model, positive, latent, audio_vae, target_audio,
                     seed, steps, scheduler, shift_video, shift_audio,
                     context_frames, video_vae=None, video_mask=None,
                     target_video=None):
        video_template, audio_template = _av_parts(latent)
        waveform, sample_rate = _validate_audio(target_audio)
        vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        if sample_rate != vae_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, vae_rate)

        context_frames = int(context_frames)
        context_video_t = _context_latent_steps(context_frames)
        if context_video_t >= video_template.shape[2]:
            raise ValueError("context is not shorter than the supplied H3 window")

        window_audio_t = int(audio_template.shape[-1])
        window_video_frames = _video_frames_from_latent_steps(
            int(video_template.shape[2])
        )
        total_audio_t = max(1, round(
            waveform.shape[-1] / vae_rate * H3_AUDIO_LATENT_FPS
        ))
        video_stride_frames = window_video_frames - context_frames
        if video_stride_frames <= 0:
            raise ValueError("continuation context is not shorter than window")

        # Derive every audio start from the absolute video-frame position.
        # 24 fps -> 40 Hz is 5/3, so a fixed rounded audio stride accumulates
        # drift. Absolute rounding naturally alternates the fractional steps.
        audio_starts_t = [0]
        while audio_starts_t[-1] + window_audio_t < total_audio_t:
            index = len(audio_starts_t)
            next_start = round(
                index * video_stride_frames
                / H3_VIDEO_FPS * H3_AUDIO_LATENT_FPS
            )
            if next_start <= audio_starts_t[-1]:
                raise ValueError("continuation window made no audio progress")
            audio_starts_t.append(next_start)

        window_samples = max(1, round(
            window_audio_t / H3_AUDIO_LATENT_FPS * vae_rate
        ))
        video_parts, audio_parts = [], []
        previous_tail = None

        for index, start_audio_t in enumerate(audio_starts_t):
            start_sample = round(
                start_audio_t / H3_AUDIO_LATENT_FPS * vae_rate
            )
            chunk = waveform[..., start_sample:start_sample + window_samples]
            chunk_audio = {"waveform": chunk, "sample_rate": vae_rate}
            sampled, _ = self.sample(
                model, positive, latent,
                (int(seed) + index) & 0xFFFFFFFFFFFFFFFF,
                steps, scheduler,
                shift_video, shift_audio, audio_vae=audio_vae,
                target_audio=chunk_audio,
                video_vae=video_vae if index == 0 else None,
                video_mask=video_mask if index == 0 else None,
                target_video=target_video if index == 0 else None,
                _context_video_latent=previous_tail,
                _context_frames=context_frames if previous_tail is not None else 0,
                _single_window=True,
            )
            window_video, window_audio = _av_parts(sampled)
            if index == 0:
                video_parts.append(window_video)
                audio_parts.append(window_audio)
            else:
                video_parts.append(window_video[:, :, context_video_t:])
                audio_stride_t = start_audio_t - audio_starts_t[index - 1]
                overlap_audio_t = window_audio_t - audio_stride_t
                if not 0 <= overlap_audio_t < window_audio_t:
                    raise ValueError("invalid audio overlap for continuation window")
                audio_parts.append(window_audio[..., overlap_audio_t:])
            previous_tail = window_video[:, :, -context_video_t:].detach().clone()

        out_video = torch.cat(video_parts, dim=2)
        wanted_frames = _align_frame_count(round(
            waveform.shape[-1] / vae_rate * H3_VIDEO_FPS
        ))
        if wanted_frames <= window_video_frames:
            wanted_video_t = _video_latent_steps(wanted_frames)
        else:
            extra_frames = wanted_frames - window_video_frames
            wanted_video_t = int(video_template.shape[2]) + math.ceil(
                extra_frames / 17
            ) * 5
        out_video = out_video[:, :, :wanted_video_t]
        out_audio = torch.cat(audio_parts, dim=-1)[..., :total_audio_t]
        output = dict(latent)
        output.pop("noise_mask", None)
        output["samples"] = comfy.nested_tensor.NestedTensor((out_video, out_audio))
        return output, dict(target_audio)


NODE_CLASS_MAPPINGS = {
    # Keep the stable workflow ID while exposing the cleaned implementation.
    "H3Ref2VAFixedAudioOneNode": MinimaxH3RefSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3Ref2VAFixedAudioOneNode": "Minimax H3 Ref Sampler",
}
