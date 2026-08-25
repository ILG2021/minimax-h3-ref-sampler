# Minimax H3 Ref Sampler — Masked Music Video

This ComfyUI custom node turns the MultiRef Music Video train into one internal
loop. It generates long MiniMax H3 Ref2VA videos against one authoritative
master soundtrack without manually wiring one sampler group per clip.

This is an experimental community project, not an official MiniMax or ComfyUI
component.

## Recommended workflow

Use the node as the complete generator and connect its finished-media outputs:

```text
Load H3 model/CLIP/VAEs ─┐
Load master audio ───────┼─> Minimax H3 Ref Sampler (Masked Music Video)
references + prompt ─────┘                         │
                                                   ├─ images ──────┐
                                                   └─ master_audio ┼─> VHS Video Combine
```

Use `images`, not a later decode of `samples`, for the final movie. Only the
`images` output contains the pixel-domain seam crossfade. The returned
`master_audio` is the original input object and is not an H3 reconstruction.

## How it works

Each H3-aligned window is prepared through ComfyUI's official
`MiniMaxH3ReferenceToVideo` node. The first window uses a normal video target.
For every continuation:

1. copy the previous sampled video latent tail into the new target prefix;
2. set that prefix's video denoise mask to `0`;
3. leave the future video mask at `1`;
4. cut the exact absolute master-audio interval for the window;
5. Audio-VAE encode it into the complete target audio latent;
6. set the complete audio denoise mask to `0`;
7. sample with native H3 AV masking;
8. decode the clip and crossfade the final matching context frames;
9. discard duplicated latent context and continue the internal loop.

Mask semantics:

```text
VIDEO: [previous tail                         ][new future ...]
MASK:  [0 0 0 0 0                            ][1 1 1 1 ...]

AUDIO: [exact master-song window                              ]
MASK:  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...              ]
```

The default context and pixel crossfade are both 39 frames. This follows the
current MultiRef Music Video workflow instead of the older guide-conditioning
Motion Context approach.

All internal windows reuse the exact same `seed`. There is no hidden
`seed + window_index` increment.

The node reports one cumulative ComfyUI progress bar across all internal
windows. Sampling steps and each window's VAE-decode/seam-assembly stage are
included, so progress does not reset to zero at every segment.

## Inputs

| Input | Description |
|---|---|
| `model`, `clip` | MiniMax H3 Ref2VA model and multimodal text encoder. |
| `video_vae`, `audio_vae` | H3 video and audio VAEs. |
| `master_audio` | Required authoritative mono/stereo soundtrack. |
| `prompt` | One H3 prompt applied to every internal window. |
| `frames` | Requested 24-fps duration, aligned upward to `5 + 17*n`; it is not inferred from audio length. |
| `window_frames` | Maximum H3 window; default/max 362. |
| `context_frames` | Protected video prefix; default 39. |
| `video_crossfade_frames` | Pixel-domain seam blend; default 39. |
| `seed` | One fixed seed reused by every internal window. |
| `steps`, `sampler_name`, `scheduler` | Sampling controls; defaults 25, `res_multistep`, `simple`. |
| `shift_video`, `shift_audio` | Native H3 AV flow shifts. |
| `ref_image_0..8` | Static image references shared by all windows. |
| `ref_video_0..2` | Reference videos sliced on the absolute timeline. |
| `ref_audio_0..2` | Reference audios sliced on the absolute timeline. |
| `ref_video_audio_0..2` | Audio paired with the corresponding reference video. |

## Outputs

| Output | Use |
|---|---|
| `samples` | Context-trimmed stitched H3 AV latent, retained for compatibility and inspection. |
| `images` | Final decoded video with pixel-domain seam crossfades. Use this for the finished movie. |
| `master_audio` | The original input waveform unchanged. Connect this directly to the final video-combine node. |

Do not decode `samples` for the final movie if you need the explicit pixel
crossfade. Connect `images` and `master_audio` to VideoHelperSuite or another
video encoder.

## Timing

H3 video uses the `5 + 17*n` frame grid. With the defaults:

```text
window 1: frames   0 .. 361
window 2: frames 323 .. 684
                    39-frame protected overlap
```

Master-audio slice endpoints are computed from absolute 24-fps frame positions,
so independently rounded clip durations cannot accumulate PCM timing drift.
The final audio output bypasses H3 Audio-VAE decoding and is therefore exactly
the supplied waveform.

For example, exactly 36 seconds at 24 fps is 864 requested frames. H3 aligns
that request upward to 872 frames. If strict picture duration matters, trim the
final encoded video externally; the original master audio itself is never
time-stretched. When the master audio is shorter than the generated picture,
the model-conditioning tail is padded with silence, but the AUDIO output still
retains the original shorter waveform.

## Requirements and memory

- current ComfyUI with official MiniMax H3 nodes and `ModelSamplingAV`;
- MiniMax H3 Ref2VA model, Video VAE, Audio VAE, and MiniMax CLIP;
- `torch` and `torchaudio`;
- batch size 1.

The node keeps generation windowed and moves completed latents to CPU. Its
`IMAGE` output must still hold the finished RGB movie in system RAM. At high
resolution or multi-minute duration, a streaming video-writer output would be
more memory efficient.

The compatibility layer probes the installed ComfyUI first. Native H3 AV-mask
support is preferred; the bundled GPL compatibility implementation activates
only when equivalent upstream capabilities are missing. A partially updated
ComfyUI mask engine is rejected instead of mixing incompatible implementations.

## Installation

Place this repository in `ComfyUI/custom_nodes`, restart ComfyUI, and add:

```text
Minimax H3 Ref Sampler (Masked Music Video)
```

After upgrading from the previous implementation, restart the whole ComfyUI
process rather than only refreshing the browser. This clears any old runtime
patches that may still exist in the current Python process.

## License and attribution

GPL-3.0. See [LICENSE](LICENSE).

The H3 AV-mask compatibility modules are derived from
[seitanism/ComfyUI-H3-Motion-Context-MultiRef](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef),
commit `87de57ba619297503fa49c9594c0c021d5b0c261`, which is a GPL-3.0
modified fork of NikoDemon80's H3 Motion Context project.
The single-node internal loop and project-specific orchestration are maintained
here under the same GPL-3.0 license.
