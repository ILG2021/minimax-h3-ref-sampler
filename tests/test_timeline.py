"""Dependency-free checks for the masked music-video timeline."""
import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "music_video.py"
NODES_SOURCE = ROOT / "nodes.py"
NAMES = {
    "H3_VIDEO_FPS",
    "H3_AUDIO_LATENT_FPS",
    "FRAME_PER_TOKEN",
    "_audio_step_at_frame",
    "_video_latent_steps",
    "_pixel_frames",
    "_align_frame_count",
    "_plan_windows",
}


def _load_helpers():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id in NAMES
                   for target in targets):
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in NAMES:
            selected.append(node)
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]),
                 str(SOURCE), "exec"), namespace)
    return namespace


H = _load_helpers()


class MusicVideoTimelineTests(unittest.TestCase):
    def test_h3_video_grid(self):
        expected_steps = {5: 2, 22: 7, 39: 12, 56: 17, 90: 27}
        for frames, steps in expected_steps.items():
            self.assertEqual(H["_video_latent_steps"](frames), steps)
            self.assertEqual(H["_pixel_frames"](steps), frames)

    def test_default_39_frame_train(self):
        self.assertEqual(
            H["_plan_windows"](685, 362, 39),
            [(0, 362), (323, 362)],
        )
        self.assertEqual(
            H["_plan_windows"](379, 362, 39),
            [(0, 362), (323, 56)],
        )

    def test_all_windows_are_phase_aligned(self):
        for total in range(5, 2500, 17):
            plan = H["_plan_windows"](total, 362, 39)
            for index, (start, length) in enumerate(plan):
                self.assertEqual(length % 17, 5)
                self.assertLessEqual(length, 362)
                if index:
                    self.assertEqual(start - plan[index - 1][0], 323)
                    self.assertGreater(length, 39)
                    source_steps = H["_video_latent_steps"](plan[index - 1][1])
                    self.assertEqual(
                        (source_steps - H["_video_latent_steps"](39)) % 5, 0
                    )

    def test_absolute_audio_stitch_has_no_grid_drift(self):
        for total in range(5, 2500, 17):
            plan = H["_plan_windows"](total, 362, 39)
            stitched = 0
            for start, length in plan:
                end = H["_audio_step_at_frame"](start + length)
                append = end - stitched
                local = H["_audio_step_at_frame"](length)
                self.assertGreater(append, 0)
                self.assertLessEqual(append, local)
                stitched = end
            self.assertEqual(
                stitched,
                H["_audio_step_at_frame"](H["_align_frame_count"](total)),
            )

    def test_masked_music_video_architecture_is_active(self):
        source = SOURCE.read_text(encoding="utf-8")
        nodes = NODES_SOURCE.read_text(encoding="utf-8")
        self.assertIn("out_video[:, :, :context_steps]", source)
        self.assertIn("video_mask[:, :, :context_steps] = 0.0", source)
        self.assertIn("audio_mask = torch.zeros(", source)
        self.assertIn('prepared["noise_mask"]', source)
        self.assertIn('denoise_mask=latent.get("noise_mask")', source)
        self.assertIn("PixelCrossfadeAssembler", nodes)
        self.assertIn("video_vae.decode(window_video)", nodes)
        self.assertIn("return output, images, master_audio", nodes)
        self.assertIn('"H3TalkingSampler": "H3 Talking Sampler"', nodes)
        self.assertNotIn("MinimaxH3RefSampler", nodes)
        self.assertIn("if not self.started:", source)
        self.assertIn("if self.tail is None or", source)
        self.assertIn("if decoded.ndim == 5:", source)
        self.assertIn("decoded = decoded.reshape(", source)
        self.assertNotIn("int(seed) + index", nodes)
        self.assertIn("ProgressBar(total_work)", nodes)
        self.assertIn("callback=progress_callback", nodes)
        self.assertIn("work_per_window = sampling_work + 1", nodes)
        self.assertEqual(nodes.count("_validate_audio"), 0)
        self.assertIn("model, positive, prepared,\n                int(seed),", nodes)
        self.assertNotIn("apply_motion_context", nodes)

    def test_project_is_gpl(self):
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("GNU GENERAL PUBLIC LICENSE", license_text)
        self.assertIn("Version 3", license_text)


if __name__ == "__main__":
    unittest.main()
