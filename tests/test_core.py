import unittest
from subtitle_project.core import (
    Word, Turn, Cue, assign_speakers, make_cues, non_speech_gaps,
    render_subtitles, timestamp,
)


class TimingTests(unittest.TestCase):
    def test_speaker_overlap_and_unknown(self):
        words = [Word(0.8, 1.4, " Hello"), Word(4, 5, " there")]
        assign_speakers(words, [Turn(0, 1, "A"), Turn(1, 2, "B")])
        self.assertEqual([w.speaker for w in words], ["B", "UNKNOWN"])

    def test_speaker_change_splits_cues(self):
        cues = make_cues([Word(0, 1, " Hi", "A"), Word(1, 2, " there", "B")])
        self.assertEqual([c.speaker for c in cues], ["A", "B"])

    def test_gap_splits_same_speaker(self):
        cues = make_cues([Word(0, 1, " Hi", "A"), Word(3, 4, " again", "A")])
        self.assertEqual(len(cues), 2)

    def test_cjk_spacing_is_preserved(self):
        cues = make_cues([Word(0, 1, "你好", "A"), Word(1, 2, "世界", "A")])
        self.assertEqual(cues[0].text, "你好世界")

    def test_subtitle_overlap_removed(self):
        cues = make_cues([Word(0, 2, " Hi", "A"), Word(1, 3, " Yes", "B")])
        self.assertLessEqual(cues[0].end, cues[1].start)

    def test_time_carry(self):
        self.assertEqual(timestamp(59.9996), "00:01:00,000")
        self.assertEqual(timestamp(3600.001, True), "01:00:00.001")

    def test_srt_vtt_and_escaping(self):
        cues = [Cue(0, 1, "A < B & C", "SPEAKER_01")]
        srt = render_subtitles(cues, "srt")
        vtt = render_subtitles(cues, "vtt")
        self.assertIn("00:00:00,000 --> 00:00:01,000", srt)
        self.assertTrue(vtt.startswith("WEBVTT\n\n"))
        self.assertIn("A &lt; B &amp; C", vtt)
        self.assertIn("[SPEAKER_01]", srt)

    def test_gap_union_and_padding(self):
        gaps = non_speech_gaps([(2, 4), (3, 5), (8, 9)], 12, 1, 0.2)
        for actual, expected in zip(gaps, [(0, 1.8), (5.2, 7.8), (9.2, 12)]):
            self.assertAlmostEqual(actual[0], expected[0])
            self.assertAlmostEqual(actual[1], expected[1])
        self.assertEqual(len(gaps), 3)

    def test_no_speech_and_all_speech(self):
        self.assertEqual(non_speech_gaps([], 10), [(0, 10)])
        self.assertEqual(non_speech_gaps([(0, 10)], 10), [])

    def test_short_gaps_not_cut(self):
        self.assertEqual(non_speech_gaps([(0, 2), (3, 5)], 5, 1, 0.2), [])

    def test_empty_outputs(self):
        self.assertEqual(render_subtitles([], "vtt"), "WEBVTT\n\n")
        self.assertEqual(render_subtitles([], "srt"), "")


if __name__ == "__main__":
    unittest.main()
