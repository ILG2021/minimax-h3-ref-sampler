"""MiniMax H3 latent-to-conditioning handoff for long-video continuity.

Adapted from AIMixer/ComfyUI_MiniMaxH3_Director's Apache-2.0 Motion Context
implementation. This version intentionally keeps only the path used by this
node: a previous AV latent becomes marked keyframe/reference conditioning for
the next fresh target latent.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .h3_context_patches import (
    CTX_AUDIO_END_KEY,
    CTX_FRAME_KEY,
    ensure_context_patches,
)

log = logging.getLogger("minimax-h3-ref-sampler.h3_motion_context")

FPS = 24.0
AUDIO_HZ = 40.0
FRAME_RESCALE = 5.0 / 3.0
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
CONTEXT_FRAME_CHOICES = (5, 22, 39, 56)
DEFAULT_CONTEXT_FRAMES = 22


def snap_context_frames(raw: int | float | None) -> int:
    """Snap a requested overlap to one supported whole H3 latent run."""
    try:
        value = int(raw or DEFAULT_CONTEXT_FRAMES)
    except (TypeError, ValueError):
        value = DEFAULT_CONTEXT_FRAMES
    return min(CONTEXT_FRAME_CHOICES, key=lambda choice: (abs(choice - value), -choice))


def pixel_frames_for_latent_t(latent_t: int) -> int:
    return sum(FRAME_PER_TOKEN[index % 5] for index in range(int(latent_t)))


def steps_for_frames(frame_count: int) -> int | None:
    steps = covered = 0
    while covered < int(frame_count):
        covered += FRAME_PER_TOKEN[steps % 5]
        steps += 1
    return steps if covered == int(frame_count) else None


def step_offsets(latent_t: int) -> list[int]:
    offsets: list[int] = []
    covered = 0
    for index in range(int(latent_t)):
        offsets.append(covered)
        covered += FRAME_PER_TOKEN[index % 5]
    return offsets


def _streams_from_latent(latent: dict) -> list[torch.Tensor]:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("Motion Context requires a MiniMax H3 AV latent")
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        streams = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        streams = list(samples)
    else:
        raise ValueError(
            "Motion Context expected a MiniMax H3 AV NestedTensor, "
            f"got {type(samples)!r}"
        )
    if len(streams) != 2:
        raise ValueError(
            f"Motion Context expected video and audio streams, got {len(streams)}"
        )
    return streams


def video_from_latent(latent: dict) -> torch.Tensor:
    video = _streams_from_latent(latent)[0]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            "Motion Context expected video latent [B,C,T,H,W], "
            f"got {tuple(video.shape)}"
        )
    return video


def _phase_aligned_tail_start(total_steps: int, context_steps: int) -> int:
    if context_steps > total_steps:
        raise ValueError(
            f"Motion Context needs {context_steps} latent steps, "
            f"but the previous window has {total_steps}"
        )
    start = int(total_steps) - int(context_steps)
    if start % 5:
        raise RuntimeError(
            "Motion Context tail is not aligned to H3's five-step temporal cycle: "
            f"start_step={start}"
        )
    return start


def _video_tail_blocks(
    latent: dict, context_frames: int
) -> tuple[list[torch.Tensor], list[int], int]:
    video = video_from_latent(latent)
    context_steps = steps_for_frames(context_frames)
    if context_steps is None:
        raise ValueError(
            f"Motion Context overlap must be one of {CONTEXT_FRAME_CHOICES}, "
            f"got {context_frames}"
        )
    start = _phase_aligned_tail_start(int(video.shape[2]), context_steps)
    blocks = [
        video[:1, :, start + index : start + index + 1].detach().clone()
        for index in range(context_steps)
    ]
    return blocks, step_offsets(context_steps), pixel_frames_for_latent_t(context_steps)


def _audio_tail_from_latent(
    latent: dict, context_frames: int
) -> tuple[torch.Tensor, int]:
    audio = _streams_from_latent(latent)[1]
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if audio.ndim != 4:
        raise ValueError(
            "Motion Context expected audio latent [B,C,2,T], "
            f"got {tuple(audio.shape)}"
        )
    context_steps = max(1, round(int(context_frames) / FPS * AUDIO_HZ))
    context_steps = min(context_steps, int(audio.shape[-1]))
    return audio[:1, ..., -context_steps:].detach().clone(), context_steps


def apply_motion_context(
    positive,
    target_latent: dict,
    *,
    context_length: int,
    context_latent: dict,
    continue_audio: bool = True,
    audio_context_length: int | None = None,
) -> tuple[Any, int]:
    """Attach the previous AV tail to the next window's conditioning.

    The target latent is inspected for shape only and is never modified.
    Returns the updated positive conditioning and pinned video frame count.
    """
    import node_helpers

    ensure_context_patches()

    context_frames = snap_context_frames(context_length)
    target_video = video_from_latent(target_latent)
    source_video = video_from_latent(context_latent)
    if target_video.shape[0] != 1 or source_video.shape[0] != 1:
        raise ValueError("Motion Context supports H3 batch size 1")
    if tuple(source_video.shape[1:2] + source_video.shape[3:]) != tuple(
        target_video.shape[1:2] + target_video.shape[3:]
    ):
        raise ValueError(
            "Motion Context source and target video layouts differ: "
            f"source={tuple(source_video.shape)}, target={tuple(target_video.shape)}"
        )

    target_frame_count = pixel_frames_for_latent_t(int(target_video.shape[2]))
    if context_frames >= target_frame_count:
        raise ValueError(
            f"Motion Context overlap {context_frames} must be shorter than "
            f"the target window {target_frame_count}"
        )

    blocks, offsets, pinned_frames = _video_tail_blocks(
        context_latent, context_frames
    )
    context_keyframes = [
        {
            "resolved_frame_index": 0,
            CTX_FRAME_KEY: int(offset),
            "latent": block,
        }
        for offset, block in zip(offsets, blocks)
    ]

    values: dict[str, Any] = {
        "minimax_keyframes": context_keyframes,
        "minimax_frame_count": target_frame_count,
    }
    output = node_helpers.conditioning_set_values(positive, values)

    if continue_audio:
        audio_frames = (
            context_frames
            if audio_context_length is None
            else max(1, int(audio_context_length))
        )
        audio_latent, audio_steps = _audio_tail_from_latent(
            context_latent, audio_frames
        )
        audio_end_coord = round(FRAME_RESCALE * pinned_frames)
        audio_reference = {
            "kind": "audio",
            "ref_audio_t": audio_steps,
            "audio_latent": audio_latent,
            CTX_AUDIO_END_KEY: audio_end_coord / FRAME_RESCALE,
        }
        output = node_helpers.conditioning_set_values(
            output, {"minimax_refs": [audio_reference]}, append=True
        )

    log.info(
        "Motion Context pinned %d video frames%s",
        pinned_frames,
        " with audio" if continue_audio else "",
    )
    return output, pinned_frames
