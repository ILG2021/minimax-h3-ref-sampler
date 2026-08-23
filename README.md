# Minimax H3 Ref Sampler

`Minimax H3 Ref Sampler` is a single-timeline ComfyUI node for MiniMax H3
Ref2VA. It combines official reference conditioning, overlapping long-video
generation, and a fixed target soundtrack in one node.

This is an experimental community node, not an official MiniMax or ComfyUI
component.

## Code structure

- `nodes.py`: public ComfyUI node, reference-conditioning loop, absolute
  timeline orchestration, and final AV stitching.
- `fixed_audio.py`: H3 frame-grid helpers, media-window slicing, sigma mapping,
  motion-context pinning, and the single-window fixed-audio sampler.
- `__init__.py`: ComfyUI node exports only.

## Features

- Builds every window through ComfyUI's official
  `MiniMaxH3ReferenceToVideo` implementation.
- Accepts up to 9 reference images, 3 reference videos, 3 reference audios,
  and 3 video-associated reference audios.
- Uses the complete `target_audio` duration as the output timeline.
- Generates at most 362 H3-aligned frames per window.
- Pins the previous window's video-latent tail into the next window at the
  current sigma, then removes the duplicated prefix during concatenation.
- Slices reference video/audio inputs and target audio on the same absolute
  24-fps timeline for every window.
- Forces the supplied target audio latent throughout sampling and ignores the
  model's predicted target-audio stream.
- Returns the original input waveform for final muxing.

The node intentionally has no local repaint, mask, multi-stage timeline,
refine pass, segment cache, or segment-export functionality.

## Fixed-audio sampling

The complete target waveform is resampled only for one Audio-VAE encode. Its
clean latent and one seed-derived global noise timeline are then sliced by
absolute 40Hz position for each window. At every sampling step:

```text
sigma_audio = map_sigma(sigma_video, shift_video, shift_audio)
audio_t = (1 - sigma_audio) * a0 + sigma_audio * epsilon_audio
```

The model sees the correctly noised target audio, but only the video prediction
is integrated. The final AV latent contains the clean target-audio latent. The
`original_audio` output bypasses resampling and the Audio VAE entirely.

## Long-video timeline

H3 video lengths use the `5 + 17*n` frame grid. The node aligns the soundtrack
duration upward to this grid and creates overlapping windows:

```text
window 1: [0 ................................ 361]
window 2:                     [340 .......... 701]
                                      22-frame context
```

The default `window_frames=362` is about 15 seconds at 24 fps. The default
`context_frames=22` is about 0.92 seconds. Supported context choices are 5, 22,
39, and 56 frames. A shortened final window is generated when possible instead
of sampling another full 362 frames and discarding most of it.

Reference images remain static. Reference videos, reference audios, and
video-associated reference audios are treated as full-timeline media and are
cut at each window's absolute start. Short media is padded: audio with silence,
video by holding its final frame.

## Inputs

| Input | Type | Description |
|---|---|---|
| `model` | `MODEL` | MiniMax H3 Ref2VA model. |
| `clip` | `CLIP` | MiniMax/Qwen3-VL text and multimodal encoder. |
| `video_vae` | `VAE` | H3 Video VAE used by official reference preparation. |
| `audio_vae` | `VAE` | H3 Audio VAE used for references and fixed target audio. |
| `target_audio` | `AUDIO` | Complete output soundtrack and duration source. |
| `prompt` | `STRING` | H3 prompt with `<Picture N>`, `<Video N>`, and `<Audio N>` references. |
| `width`, `height` | `INT` | Output canvas passed to official conditioning. |
| `window_frames` | `INT` | Maximum aligned generation window; default/max 362. |
| `context_frames` | choice | Video continuation overlap: 5, 22, 39, or 56. |
| `ref_image_size` | choice | Official reference-image sizing mode. |
| `seed` | `INT` | Base seed; window `N` uses `seed + N`. |
| `steps`, `scheduler` | sampling | H3 sampling schedule. |
| `shift_video`, `shift_audio` | `FLOAT` | H3 video/audio flow shifts. |
| `ref_image_0..8` | `IMAGE` | Static picture references. |
| `ref_video_0..2` | `IMAGE` | Timeline-aligned reference videos. |
| `ref_audio_0..2` | `AUDIO` | Timeline-aligned reference audios. |
| `ref_video_audio_0..2` | `AUDIO` | Audio paired with the corresponding reference video. |

## Outputs

| Output | Type | Description |
|---|---|---|
| `samples` | `LATENT` | Stitched H3 AV latent containing generated video and clean fixed-audio latent. |
| `original_audio` | `AUDIO` | Original `target_audio`; connect directly to the final video muxer. |

## Installation

Place this repository under `ComfyUI/custom_nodes`, restart ComfyUI, and search
for:

```text
Minimax H3 Ref Sampler
```

Requirements:

- a current ComfyUI build with official MiniMax H3 nodes;
- MiniMax H3 Ref2VA model, Video VAE, Audio VAE, and MiniMax CLIP;
- the ComfyUI environment's `torch` and `torchaudio`;
- batch size 1.

## Important behavior

- Final video duration is H3-grid aligned and can be up to 16 frames longer
  than the exact soundtrack duration. Use `original_audio` for muxing and trim
  video at export if exact waveform duration is required.
- Extremely long stitched AV latents can still consume substantial memory at
  decode time even though sampling itself is windowed.
- `ref_image_size=max`, long reference media, and high output resolution can
  substantially increase memory use.
- The node expects the official H3 AV layout: video `[B,24,T,H,W]`, audio
  `[B,32,2,T]`.

## Attribution

The single-timeline implementation uses the long-video planning ideas and
official-conditioning integration patterns from
[AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director),
licensed under Apache-2.0. Fixed-target-audio sampling is implemented in this
project.
