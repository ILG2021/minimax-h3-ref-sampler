"""Dependency-free checks for H3 sliding-window timeline arithmetic."""
import ast
import unittest
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "fixed_audio.py"
NAMES = {
    "H3_VIDEO_FPS",
    "H3_AUDIO_LATENT_FPS",
    "_audio_step_at_frame",
    "_audio_overlap_range",
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

    def test_audio_overlap_uses_absolute_boundaries(self):
        first = H["_audio_overlap_range"](340, 22)
        second = H["_audio_overlap_range"](680, 22)
        self.assertEqual(first, (567, 603))
        self.assertEqual(second, (1133, 1170))
        self.assertEqual(first[1] - first[0], 36)
        self.assertEqual(second[1] - second[0], 37)

    def test_every_continuation_is_covered_by_previous_window(self):
        for context in (5, 22, 39, 56):
            plan = H["_plan_windows"](2079, 362, context)
            for (previous_start, previous_length), (start, _) in zip(plan, plan[1:]):
                overlap_start, overlap_end = H["_audio_overlap_range"](
                    start, context
                )
                previous_audio_start = H["_audio_step_at_frame"](previous_start)
                previous_audio_end = H["_audio_step_at_frame"](
                    previous_start + previous_length
                )
                self.assertGreaterEqual(overlap_start, previous_audio_start)
                self.assertLessEqual(overlap_end, previous_audio_end)

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


if __name__ == "__main__":
    unittest.main()
