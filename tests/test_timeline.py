"""Dependency-free checks for H3 sliding-window timeline arithmetic."""
import ast
import unittest
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "fixed_audio.py"
NAMES = {
    "H3_VIDEO_FPS",
    "H3_AUDIO_LATENT_FPS",
    "_audio_step_at_frame",
    "_video_latent_steps",
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
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"),
         namespace)
    return namespace


H = _load_helpers()


class SlidingTimelineTests(unittest.TestCase):
    def test_full_and_short_final_windows(self):
        self.assertEqual(
            H["_plan_windows"](702, 362, 22),
            [(0, 362), (340, 362)],
        )
        self.assertEqual(
            H["_plan_windows"](379, 362, 22),
            [(0, 362), (340, 39)],
        )

    def test_audio_overlap_matches_discarded_local_prefix(self):
        for context in (5, 22, 39, 56):
            plan = H["_plan_windows"](2079, 362, context)
            stitched_end = H["_audio_step_at_frame"](plan[0][1])
            for start, length in plan[1:]:
                local_window = H["_audio_step_at_frame"](length)
                absolute_end = H["_audio_step_at_frame"](start + length)
                append = absolute_end - stitched_end
                overlap = local_window - append
                self.assertGreater(overlap, 0)
                self.assertLess(overlap, local_window)
                self.assertEqual(overlap + append, local_window)
                local_start = absolute_end - local_window
                self.assertEqual(local_start + overlap, stitched_end)
                stitched_end = absolute_end

    def test_stitched_video_steps_match_aligned_timeline(self):
        for total in (362, 379, 702, 1042, 2079):
            plan = H["_plan_windows"](total, 362, 22)
            context_steps = H["_video_latent_steps"](22)
            stitched = H["_video_latent_steps"](plan[0][1])
            stitched += sum(
                H["_video_latent_steps"](length) - context_steps
                for _, length in plan[1:]
            )
            self.assertEqual(stitched, H["_video_latent_steps"](
                H["_align_frame_count"](total)
            ))

    def test_window_properties_across_supported_grid(self):
        for window in range(22, 363, 17):
            for context in (5, 22, 39, 56):
                if context >= window:
                    continue
                for requested in range(5, 2048, 17):
                    total = H["_align_frame_count"](requested)
                    plan = H["_plan_windows"](total, window, context)
                    self.assertEqual(plan[0][0], 0)
                    self.assertEqual(plan[-1][0] + plan[-1][1], total)
                    for index, (start, length) in enumerate(plan):
                        self.assertLessEqual(length, window)
                        self.assertEqual(length % 17, 5)
                        if index:
                            self.assertEqual(
                                start - plan[index - 1][0], window - context
                            )
                            self.assertGreater(length, context)


if __name__ == "__main__":
    unittest.main()
