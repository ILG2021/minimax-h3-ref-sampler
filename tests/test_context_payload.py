"""Dependency-free checks for the process-wide H3 payload wrapper."""
import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "h3_context_patches.py"


class _Cond:
    def __init__(self, payload):
        self.cond = payload


class _MiniMaxH3:
    def extra_conds(self, **kwargs):
        payload = {}
        keyframes = kwargs.get("minimax_keyframes")
        refs = kwargs.get("minimax_refs")
        if keyframes is not None:
            payload["cond_video_latents"] = [
                item["latent"] for item in keyframes if "latent" in item
            ]
            payload["cond_audio_latents"] = [
                item["audio_latent"]
                for item in keyframes
                if item.get("audio_latent") is not None
            ]
        if refs is not None:
            # Reproduce older ComfyUI's overwrite that the wrapper repairs.
            payload["cond_video_latents"] = [
                item["latent"] for item in refs if "latent" in item
            ]
            payload["cond_audio_latents"] = [
                item["audio_latent"]
                for item in refs
                if item.get("audio_latent") is not None
            ]
        return {"minimax_payload": _Cond(payload)}


class _PackedLayout:
    def __init__(self, *args, **kwargs):
        del args, kwargs


@contextmanager
def _loaded_patch():
    saved = {
        name: sys.modules.get(name)
        for name in (
            "torch",
            "comfy",
            "comfy.model_base",
            "comfy.ldm",
            "comfy.ldm.minimax",
            "comfy.ldm.minimax.model",
        )
    }
    fake_comfy = types.ModuleType("comfy")
    fake_model_base = types.ModuleType("comfy.model_base")
    fake_ldm = types.ModuleType("comfy.ldm")
    fake_minimax = types.ModuleType("comfy.ldm.minimax")
    fake_minimax_model = types.ModuleType("comfy.ldm.minimax.model")
    fake_model_base.MiniMaxH3 = _MiniMaxH3
    fake_minimax_model.PackedLayout = _PackedLayout
    fake_minimax_model.FRAME_RESCALE = 5.0 / 3.0
    fake_minimax.model = fake_minimax_model
    fake_ldm.minimax = fake_minimax
    fake_comfy.model_base = fake_model_base
    fake_comfy.ldm = fake_ldm
    sys.modules["torch"] = types.ModuleType("torch")
    sys.modules["comfy"] = fake_comfy
    sys.modules["comfy.model_base"] = fake_model_base
    sys.modules["comfy.ldm"] = fake_ldm
    sys.modules["comfy.ldm.minimax"] = fake_minimax
    sys.modules["comfy.ldm.minimax.model"] = fake_minimax_model
    name = "_h3_context_patches_test"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop(name, None)
        for module_name, previous in saved.items():
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous
        _MiniMaxH3.extra_conds = _ORIGINAL_EXTRA_CONDS
        _PackedLayout.__init__ = _ORIGINAL_LAYOUT_INIT


_ORIGINAL_EXTRA_CONDS = _MiniMaxH3.extra_conds
_ORIGINAL_LAYOUT_INIT = _PackedLayout.__init__


class PayloadPatchTests(unittest.TestCase):
    def test_marked_context_merges_keyframes_and_refs_in_layout_order(self):
        with _loaded_patch() as patch:
            patch._payload_orig = _MiniMaxH3.extra_conds
            keyframes = [{
                "latent": "context-video",
                "audio_latent": "context-guide-audio",
                patch.CTX_FRAME_KEY: 0,
            }]
            refs = [
                {"kind": "image", "latent": "reference-image"},
                {"kind": "audio", "audio_latent": "context-audio"},
            ]
            payload = patch._context_extra_conds(
                _MiniMaxH3(),
                minimax_keyframes=keyframes,
                minimax_refs=refs,
                minimax_frame_count=362,
            )["minimax_payload"].cond
            self.assertEqual(
                payload["cond_video_latents"],
                ["context-video", "reference-image"],
            )
            self.assertEqual(
                payload["cond_audio_latents"],
                ["context-guide-audio", "context-audio"],
            )
            self.assertEqual(payload["frame_count"], 362)

    def test_unmarked_stock_graph_is_unchanged(self):
        with _loaded_patch() as patch:
            patch._payload_orig = _MiniMaxH3.extra_conds
            payload = patch._context_extra_conds(
                _MiniMaxH3(),
                minimax_keyframes=[{"latent": "stock-keyframe"}],
                minimax_refs=[{"kind": "image", "latent": "stock-reference"}],
            )["minimax_payload"].cond
            self.assertEqual(payload["cond_video_latents"], ["stock-reference"])

    def test_replacement_after_install_is_detected(self):
        with _loaded_patch() as patch:
            self.assertTrue(patch.ensure_payload_patch())

            def foreign(self, **kwargs):
                return _ORIGINAL_EXTRA_CONDS(self, **kwargs)

            setattr(foreign, "_h3_motion_context_payload_patch", True)
            _MiniMaxH3.extra_conds = foreign
            with self.assertRaisesRegex(RuntimeError, "already patched"):
                patch.ensure_payload_patch()

    def test_preflight_does_not_partially_install_on_payload_conflict(self):
        with _loaded_patch() as patch:
            original_layout = _PackedLayout.__init__

            def foreign(self, **kwargs):
                return _ORIGINAL_EXTRA_CONDS(self, **kwargs)

            setattr(foreign, "_h3_motion_context_payload_patch", True)
            _MiniMaxH3.extra_conds = foreign
            with self.assertRaisesRegex(RuntimeError, "cannot start"):
                patch.ensure_context_patches()
            self.assertIs(_PackedLayout.__init__, original_layout)


if __name__ == "__main__":
    unittest.main()
