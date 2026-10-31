"""Run from the project root: python -m unittest discover tests"""

import unittest

from chords import find_chords, group_by_onset, is_chord, name_chord
from postprocess import NoteEvent


def note(midi, start, stop=1.0):
    return NoteEvent(start=start, stop=stop, midi=midi, amplitude=0.7)


class NameChordTest(unittest.TestCase):
    def test_seventh_chords(self):
        self.assertEqual(name_chord([50, 53, 57, 60]), "Dm7")
        self.assertEqual(name_chord([43, 59, 62, 65]), "G7")
        self.assertEqual(name_chord([48, 52, 55, 59]), "Cmaj7")

    def test_jazz_voicings(self):
        self.assertEqual(name_chord([50, 65, 76]), "Dm9")        # rootless-style D F E
        self.assertEqual(name_chord([43, 65, 69, 76]), "G13")    # G F A E
        self.assertEqual(name_chord([48, 67, 71, 74]), "Cmaj9")  # C G B D

    def test_inversion_gets_slash_bass(self):
        self.assertEqual(name_chord([52, 55, 60]), "C/E")

    def test_single_notes_and_dyads(self):
        self.assertEqual(name_chord([60]), "C4")
        self.assertEqual(name_chord([50, 65]), "D3+F4")


class GroupingTest(unittest.TestCase):
    def test_rolled_chord_is_one_group(self):
        notes = [note(55, 9.276), note(69, 9.296), note(65, 9.244), note(76, 9.395)]
        self.assertEqual(len(group_by_onset(notes)), 1)

    def test_separate_chords_split(self):
        notes = [note(50, 0.57), note(65, 0.58), note(55, 2.35), note(65, 2.36)]
        self.assertEqual(len(group_by_onset(notes)), 2)

    def test_slow_arpeggio_is_capped_by_span(self):
        notes = [note(48 + i, i * 0.1) for i in range(6)]  # 100 ms apart, 500 ms total
        self.assertGreater(len(group_by_onset(notes)), 1)

    def test_find_chords_counts_ii_v_i(self):
        notes = [note(50, 0.57), note(65, 0.58), note(76, 0.57),
                 note(43, 2.35), note(65, 2.36), note(69, 2.35), note(76, 2.35),
                 note(48, 4.29), note(67, 4.31), note(71, 4.30), note(74, 4.31)]
        chords = find_chords(notes)
        self.assertEqual([c.name for c in chords], ["Dm9", "G13", "Cmaj9"])
        self.assertTrue(all(is_chord(c) for c in chords))


if __name__ == "__main__":
    unittest.main()
