"""Run from the project root: python -m unittest discover tests"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import pipeline
from cnn_filter import (BELOW, BINS_PER_SEMITONE, CQT_SEMITONES, FRAMES, PITCH_BINS, PITCH_JITTER, TIME_JITTER,
                        build_model, centre, cut_patches)
from eval.maps import _in_split, _piano_and_subset
from features import FEATURE_NAMES
from postprocess import NoteEvent


def note(midi, start):
    return NoteEvent(start=start, stop=start + 0.5, midi=midi, amplitude=0.7)


class PatchTest(unittest.TestCase):
    def setUp(self):
        self.cqt = np.random.default_rng(0).random((CQT_SEMITONES * BINS_PER_SEMITONE, 400)).astype(np.float32)

    def test_shapes(self):
        patches = cut_patches(self.cqt, [note(60, 1.0), note(21, 0.0), note(108, 4.5)])
        self.assertEqual(patches.shape, (3, PITCH_BINS + 2 * PITCH_JITTER, FRAMES + 2 * TIME_JITTER))
        self.assertEqual(centre(patches).shape, (3, PITCH_BINS, FRAMES))

    def test_note_sits_at_the_same_row(self):
        # The note's own bin is always BELOW semitones up from the bottom of the centred patch.
        cqt = np.zeros_like(self.cqt)
        for midi in (21, 60, 108):
            cqt[:] = 0
            cqt[(midi - 21) * BINS_PER_SEMITONE, :] = 1
            patch = centre(cut_patches(cqt, [note(midi, 1.0)]))[0]
            self.assertEqual(int(patch[:, 10].argmax()), BELOW * BINS_PER_SEMITONE + 1)

    def test_edges_are_zero_padded(self):
        patch = centre(cut_patches(self.cqt, [note(21, 0.0)]))[0]
        self.assertTrue(np.all(patch[:BELOW * BINS_PER_SEMITONE] == 0))   # below A0
        self.assertTrue(np.all(patch[:, :5] == 0))                         # before the audio starts


class ModelTest(unittest.TestCase):
    def test_one_logit_per_note(self):
        import torch

        model = build_model().eval()
        logits = model(torch.zeros(4, PITCH_BINS, FRAMES), torch.zeros(4, len(FEATURE_NAMES)))
        self.assertEqual(tuple(logits.shape), (4,))


class AutoModeTest(unittest.TestCase):
    def test_auto_prefers_cnn(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "MODELS_DIR", Path(tmp)):
            self.assertEqual(pipeline.resolve_mode("auto", "piano"), "default")
            pipeline.learned_model_path("piano").touch()
            self.assertEqual(pipeline.resolve_mode("auto", "piano"), "learned")
            pipeline.cnn_model_path("piano").touch()
            self.assertEqual(pipeline.resolve_mode("auto", "piano"), "cnn")


class SplitTest(unittest.TestCase):
    def split_of(self, stem, scheme="enstdk"):
        piano, subset = _piano_and_subset(stem)
        return next(s for s in ("train", "val", "test") if _in_split(piano, subset, s, scheme))

    def test_enstdk(self):
        self.assertEqual(self.split_of("MAPS_UCHO_C0-4-7_I32-96_S0_n1_ENSTDkAm"), "train")
        self.assertEqual(self.split_of("MAPS_MUS-alb_se3_ENSTDkAm"), "train")
        self.assertEqual(self.split_of("MAPS_RAND_P1_M_I32-96_S0_n1_ENSTDkCl"), "val")
        self.assertEqual(self.split_of("MAPS_MUS-alb_se3_ENSTDkCl"), "test")

    def test_maps_full(self):
        self.assertEqual(self.split_of("MAPS_MUS-alb_se3_AkPnBcht", "maps-full"), "train")
        self.assertEqual(self.split_of("MAPS_MUS-alb_se3_StbgTGd2", "maps-full"), "val")
        self.assertEqual(self.split_of("MAPS_MUS-alb_se3_ENSTDkAm", "maps-full"), "test")


if __name__ == "__main__":
    unittest.main()
