# Minimax H3 Ref Sampler

A single ComfyUI sampling node for **MiniMax H3 Ref2VA** that uses a supplied
soundtrack as the target audio trajectory while generating the video stream.

The node is intended for audio-driven portrait and dialogue generation where:

- the generated mouth motion should follow an existing soundtrack;
- H3 should see that soundtrack at the correct audio noise level on every step;
- the model must not rewrite the final soundtrack;
- the original ComfyUI sampling block should be replaced by one node.

> [!IMPORTANT]
> This is an experimental sampler, not an official MiniMax or ComfyUI node.
> It preserves the final waveform only when the `original_audio` output is used
> for the final video mux. Audio-VAE decoding is not waveform-lossless.

## What it replaces

`Minimax H3 Ref Sampler` replaces this part of the standard workflow:

```text
RandomNoise
Basic Guider
KSamplerSelect
BasicScheduler
SamplerCustomAdvanced
```

Keep the built-in `MiniMax H3 Reference to Video` node. It still prepares the
prompt, Qwen multimodal conditioning, reference latents, and empty target AV
latent. This custom node takes over only the sampling stage and target-audio
preparation.

## How it works

The supplied waveform is resampled to the H3 Audio VAE rate when necessary and
encoded once as clean target audio latent `a0`. One fixed audio-noise realization
`epsilon_audio` is created from `seed`.

At each video sampling step, the node maps the current video sigma to H3's audio
clock and reconstructs the target audio state:

```text
sigma_audio = map_sigma(sigma_video, shift_video, shift_audio)

a_t = (1 - sigma_audio) * a0
    + sigma_audio * epsilon_audio
```

H3 jointly receives the current target video and `a_t`, together with the
`positive` conditioning produced by the built-in Ref2VA node. The model predicts
both video and audio, but this sampler:

1. integrates only the video prediction;
2. discards the model-predicted target audio;
3. reconstructs audio again from `a0`, fixed noise, and the next audio sigma;
4. forces the final audio latent back to `a0`.

```text
fixed prompt and reference conditions
                  |
                  v
current video + sigma-matched target audio --> H3
                  |                            |
                  |                       AV prediction
                  |                            |
                  +---- update video only <----+

next audio = rebuild(a0, fixed noise, next sigma_audio)
```

## Requirements

- A current ComfyUI build with native MiniMax H3 support.
- MiniMax H3 **Ref2VA** model.
- MiniMax H3 Video VAE and Audio VAE.
- `torch` and `torchaudio` from the ComfyUI environment.
- Batch size `1`.

The node expects the native H3 AV latent layout:

```text
video: [B, 24, T_video, H, W]
audio: [B, 32, 2, T_audio]
```

## Installation

Clone the repository into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone <YOUR_GITHUB_REPOSITORY_URL> comfyui_h3_fixed_audio
```

Or copy the `comfyui_h3_fixed_audio` directory into:

```text
ComfyUI/custom_nodes/comfyui_h3_fixed_audio
```

Restart ComfyUI, then search for:

```text
Minimax H3 Ref Sampler
```

The node is located under:

```text
sampling/minimax
```

## Long videos

`Minimax H3 Ref Sampler` automatically switches to sequential overlapping
windows when `target_audio` is longer than the connected native H3 latent.
For long-form work, supply a maximum trained window (362 aligned frames, about
15 seconds). No separate long-video node or external loop is required.

The first window is sampled normally. Every later window pins the previous
window's tail in latent space, generates the remaining area, and discards the
duplicated head before concatenation. The default `context_frames=22` carries
about 0.92 seconds of motion without a Video-VAE decode/encode round trip.
When repaint inputs are connected, repaint is applied to the first window;
later windows continue from that result through the same latent overlap.

Supported continuation spans are `5`, `22`, `39`, and `56` frames because they
land exactly on H3's temporal latent grid. Each window uses `seed + window_index`.
The original target waveform is returned for final muxing; window audio is cut
against one absolute 40 Hz latent timeline.

For inputs no longer than one window, the node performs one sampling pass.
The output is a combined AV latent, so decoding an extremely long result can
still require substantial host/GPU memory even though sampling remains
window-sized.

## Inputs

| Input | Type | Recommended source | Description |
|---|---|---|---|
| `model` | `MODEL` | `Get_REF2VA.model` | MiniMax H3 Ref2VA model. The node clones and patches its sampling configuration. |
| `positive` | `CONDITIONING` | `MiniMax H3 Reference to Video.positive` | Prompt, Qwen hidden states, modality tags, and `minimax_refs`. |
| `latent` | `LATENT` | `MiniMax H3 Reference to Video.LATENT` | Native joint target video/audio latent. |
| `audio_vae` | `VAE` | `Get_Audio_VAE` | MiniMax H3 Audio VAE. Do not connect the Video VAE. |
| `target_audio` | `AUDIO` | `Load Audio` or video loader audio output | Soundtrack used as the target audio trajectory and returned unchanged. |
| `seed` | `INT` | Widget | Controls video noise and the fixed target-audio noise realization. |
| `steps` | `INT` | Widget | Number of requested sampling steps. Default: `8`. |
| `scheduler` | choice | Widget | Native ComfyUI scheduler. Default: `beta`, matching the shown standard workflow. |
| `shift_video` | `FLOAT` | Widget | H3 video flow shift. Default: `12.0`. |
| `shift_audio` | `FLOAT` | Widget | H3 audio flow shift. Default: `3.0`. |

Do not change the shift values unless the model/workflow you are using expects
different H3 sampling parameters.

## Outputs

| Output | Type | Connect to | Description |
|---|---|---|---|
| `samples` | `LATENT` | `VAE Decode.samples` | Final H3 AV latent: generated video plus clean target audio latent. |
| `original_audio` | `AUDIO` | `Video Combine.audio` | The original input `AUDIO`, bypassing Audio-VAE reconstruction. |

## Workflow migration

Remove or bypass:

```text
RandomNoise
Basic Guider
KSamplerSelect
BasicScheduler
SamplerCustomAdvanced
VAE Decode Audio   # not needed for final mux
```

Keep:

```text
Get_REF2VA
MiniMax H3 Reference to Video
Get_Audio_VAE
Get_Video_VAE
VAE Decode
Video Combine
your audio/video/reference loaders
```

Connect the new node as follows:

```text
Get_REF2VA.model
    -> Minimax H3 Ref Sampler.model

MiniMax H3 Reference to Video.positive
    -> Minimax H3 Ref Sampler.positive

MiniMax H3 Reference to Video.LATENT
    -> Minimax H3 Ref Sampler.latent

Get_Audio_VAE
    -> Minimax H3 Ref Sampler.audio_vae

Load Audio.AUDIO
    -> Minimax H3 Ref Sampler.target_audio

Minimax H3 Ref Sampler.samples
    -> VAE Decode.samples

Get_Video_VAE
    -> VAE Decode.vae

VAE Decode.IMAGE
    -> Video Combine.images

Minimax H3 Ref Sampler.original_audio
    -> Video Combine.audio
```

Compact view:

```text
Get_REF2VA.model ----------------------------------+
                                                    |
H3 Reference to Video.positive --------------------|
H3 Reference to Video.LATENT ----------------------|--> Minimax H3 Ref Sampler
H3 Audio VAE --------------------------------------|            |
Target AUDIO --------------------------------------+            +--> samples
                                                                 |      |
                                                                 |   Video VAE Decode
                                                                 |      |
                                                                 |   video frames
                                                                 |
                                                                 +--> original_audio
                                                                        |
                                                      Video Combine <---+
```

## Reference audio versus target audio

The two paths have different meanings:

```text
ref_audio_0 / ref_video_audio_0
    = fixed Ref2VA conditioning stored in positive/minimax_refs

target_audio
    = the target audio state reconstructed at every sampling sigma
```

For stronger semantic alignment, the same soundtrack can be connected to both
paths:

```text
Load Audio.AUDIO
    +--> MiniMax H3 Reference to Video.ref_audio_0
    +--> Minimax H3 Ref Sampler.target_audio
```

For a reference video's own soundtrack:

```text
Video Loader.AUDIO
    +--> MiniMax H3 Reference to Video.ref_video_audio_0
    +--> Minimax H3 Ref Sampler.target_audio
```

The first connection provides reference semantics. The second controls the
target diffusion trajectory and drives the generated mouth motion.

## Audio duration behavior

The input H3 AV latent defines one sampling-window duration:

- longer target audio automatically activates overlapping window generation;
- shorter target audio is zero-padded in latent space;
- `original_audio` itself is returned unchanged.

H3 video frames must land on its `5 + 17*n` temporal grid. The final generated
picture can therefore extend slightly beyond an arbitrary waveform duration;
the untouched `original_audio` remains the mux source.

## Recommended starting settings

Use the values from the standard workflow first:

```text
steps        = 8
scheduler    = beta
shift_video  = 12.0
shift_audio  = 3.0
```

Keep `seed` fixed while comparing prompts or reference inputs. Increasing steps
may improve detail on a non-distilled model, but it does not guarantee more
accurate lip sync. Use the step count expected by the exact checkpoint or Turbo
LoRA configuration.

## Why `original_audio` should be used

The node returns a clean Audio-VAE latent as part of `samples`, so `VAE Decode
Audio` can still be used for debugging. However:

```text
waveform -> Audio VAE encode -> Audio VAE decode
```

is lossy. It may change high-frequency detail, phase, loudness, or other waveform
properties. For an unchanged final soundtrack, always connect:

```text
original_audio -> Video Combine.audio
```

This keeps the original ComfyUI `AUDIO` object out of the model's output path.

## Limitations

- This node currently supports H3 batch size `1` only.
- The sampler is Euler-based internally; arbitrary samplers are not exposed.
- Existing `noise_mask` data on the input latent is intentionally removed. A
  generic inpaint mask uses the outer video clock and can overwrite the custom
  sigma-consistent audio trajectory.
- The node requires a native H3 joint AV latent. It is not a general audio-driven
  video sampler for other architectures.
- Exact waveform preservation does not itself guarantee perfect lip sync. H3
  still decides how the visual stream responds to target audio and references.
- Multiple speakers, occluded mouths, profile faces, rapid dialogue, very small
  faces, or weak identity references may reduce visual sync quality.
- This implementation has been statically reviewed against current ComfyUI H3
  sampling interfaces, but users should test it with their exact ComfyUI commit,
  checkpoint, quantization, and acceleration patches.

## Troubleshooting

### Node does not appear

- Confirm the folder is directly under `ComfyUI/custom_nodes`.
- Confirm it contains `__init__.py` and `nodes.py`.
- Restart ComfyUI completely.
- Check the terminal for Python import errors.

### `Expected H3 channels` or AV layout error

Connect the `LATENT` output from the built-in `MiniMax H3 Reference to Video`
node. Do not connect an image-only or non-H3 latent.

### Audio VAE layout mismatch

The `audio_vae` input received the wrong VAE. Connect the MiniMax H3 Audio VAE,
not the H3 Video VAE.

### Final audio is still changed

Check that `Video Combine.audio` receives this node's `original_audio` output.
If it receives `VAE Decode Audio`, the waveform has gone through lossy VAE
reconstruction.

### Mouth timing is offset

- Make sure `target_audio` starts at the intended video time; remove leading
  silence if it is accidental.
- Ensure the reference video frames are supplied at the workflow's expected FPS.
- Align/crop audio duration before the sampler.
- Keep the same soundtrack connected as both reference and target audio when
  appropriate.
- Verify that the final video-combine node is not independently trimming audio.

### Output quality differs from the original workflow

- Match `steps`, `scheduler`, `shift_video`, `shift_audio`, and `seed`.
- Confirm that the checkpoint or Turbo LoRA supports the selected step count.
- Temporarily disable unrelated sampler/model patches when comparing results.

### Out of memory

The sampler still runs the full H3 Ref2VA Transformer. Reduce reference sizes,
reference-video length, generation resolution, or output duration. In the
built-in Ref2VA node, `ref_image_size=max` can substantially increase reference
token count and memory use.

## Implementation notes

- Reference latents remain in `positive/minimax_refs`; they are not denoised.
- ComfyUI packs target video and target audio into one flattened sampler tensor.
- The custom sampler replaces only the target-audio part before every model call.
- H3 receives target audio on its mapped audio clock.
- Model-predicted target audio is ignored.
- Final target audio latent is explicitly replaced by `a0`.
- The model object is cloned before its sampling configuration is patched.
- The input model and conditioning objects are not mutated in place.

Core step, simplified:

```python
for sigma_v, sigma_v_next in sigma_pairs:
    sigma_a = map_video_sigma_to_audio_sigma(sigma_v)
    sigma_a_next = map_video_sigma_to_audio_sigma(sigma_v_next)

    audio_t = (1 - sigma_a) * clean_audio + sigma_a * fixed_audio_noise
    prediction = h3(video_t, audio_t, positive)

    video_t = euler_update(video_t, prediction.video, sigma_v, sigma_v_next)
    audio_next = (
        (1 - sigma_a_next) * clean_audio
        + sigma_a_next * fixed_audio_noise
    )

return video_t, clean_audio
```

## License and attribution

Before publishing, add your chosen license file and repository URL. This node
integrates with ComfyUI's public extension interfaces and is designed around the
native MiniMax H3 node and sampling contracts. MiniMax, H3, and ComfyUI are
trademarks or projects of their respective owners; this repository is not an
official release from them.
