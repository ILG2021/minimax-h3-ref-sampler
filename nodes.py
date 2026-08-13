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
    video_values: int,
    video_shift: float,
    audio_shift: float,
):
    """Build Euler sampler that integrates video only and reconstructs audio."""
    packed_values = video_values + math.prod(clean_audio.shape[1:])
    clean_cpu = clean_audio.detach().to(device="cpu", dtype=torch.float32)
    noise_cpu = audio_noise.detach().to(device="cpu", dtype=torch.float32)

    def sample_fixed_audio(model_wrap, x, sigmas, extra_args=None, callback=None,
                           disable=None):
        if x.ndim != 2 or x.shape[-1] != packed_values:
            raise ValueError(
                f"packed H3 latent mismatch: expected [B,{packed_values}], "
                f"got {tuple(x.shape)}"
            )
        args = {} if extra_args is None else extra_args
        clean = clean_cpu.to(device=x.device, dtype=x.dtype).reshape(x.shape[0], -1)
        noise = noise_cpu.to(device=x.device, dtype=x.dtype).reshape(x.shape[0], -1)
        sigma_batch = x.new_ones([x.shape[0]])
        video = x[..., :video_values]

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
                "audio_vae": ("VAE",),
                "target_audio": ("AUDIO",),
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
            }
        }

    RETURN_TYPES = ("LATENT", "AUDIO")
    RETURN_NAMES = ("samples", "original_audio")
    FUNCTION = "sample"
    CATEGORY = "sampling/minimax"

    def sample(self, model, positive, latent, audio_vae, target_audio, seed,
               steps, scheduler, shift_video, shift_audio):
        video, audio_template = _av_parts(latent)
        if video.shape[0] != 1:
            raise ValueError("MiniMax H3 Ref Sampler supports batch size 1")

        waveform, sample_rate = _validate_audio(target_audio)
        vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        if sample_rate != vae_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, vae_rate
            )
        clean_audio = audio_vae.encode(waveform.movedim(1, -1))
        clean_audio = _fit_audio_latent(clean_audio, audio_template)

        # Do not pass an inherited inpaint mask into model_wrap. Its generic
        # video-clock blending could overwrite sigma-consistent target audio.
        prepared = dict(latent)
        prepared.pop("noise_mask", None)
        prepared["samples"] = comfy.nested_tensor.NestedTensor(
            (video, clean_audio)
        )

        sampling = _make_model_sampling(model, shift_video)
        patched_model = _patch_h3_model(
            model, sampling, shift_video, shift_audio
        )
        sigmas = comfy.samplers.calculate_sigmas(
            sampling, scheduler, int(steps)
        ).cpu()
        # Some discrete schedulers legitimately collapse duplicate timesteps,
        # so the resulting list may be shorter than steps+1.
        if len(sigmas) < 2 or not torch.isclose(
            sigmas[-1], sigmas.new_tensor(0.0)
        ):
            raise ValueError("scheduler must return at least two sigmas ending at zero")

        # Dedicated fixed audio epsilon_a. It is generated once and reused for
        # every step. Video noise is generated by ComfyUI below with the seed.
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        audio_noise = torch.randn(
            clean_audio.shape, generator=generator, dtype=torch.float32
        )
        audio_noise *= float(getattr(sampling, "noise_scale", 1.0))
        sampler = _make_fixed_audio_sampler(
            clean_audio,
            audio_noise,
            math.prod(video.shape[1:]),
            shift_video,
            shift_audio,
        )

        latent_samples = prepared["samples"]
        noise = comfy.sample.prepare_noise(
            latent_samples, int(seed), prepared.get("batch_index")
        )
        guider = _BasicPositiveGuider(patched_model)
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
        return output, target_audio


NODE_CLASS_MAPPINGS = {
    # Keep the stable workflow ID while exposing the cleaned implementation.
    "H3Ref2VAFixedAudioOneNode": MinimaxH3RefSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3Ref2VAFixedAudioOneNode": "Minimax H3 Ref Sampler",
}
