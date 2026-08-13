import unittest

from scripts.summarize_selected_tags import (
    format_background_block,
    format_selection_block,
)


class SummarizeSelectedTagsTest(unittest.TestCase):
    def test_formats_signal_as_one_interval_per_line(self):
        lines = format_selection_block(
            "signal_selection",
            "Bpt > 10 and Bpt < 50 and abs(By) < 1.6 and BQvalue < 0.15",
        )
        self.assertEqual(
            lines,
            [
                "signal_selection:",
                "  Bpt:[10,50]",
                "  By:[-1.6,1.6]",
                "  BQvalue:[-inf,0.15]",
            ],
        )

    def test_background_omits_shared_signal_intervals(self):
        signal = "Bpt > 10 and Bpt < 50 and abs(By) < 1.6"
        background = (
            "((Bmass > 3.82 and Bmass < 3.85) or "
            "(Bmass > 3.89 and Bmass < 3.92)) and "
            "Bpt > 10 and Bpt < 50 and abs(By) < 1.6"
        )
        self.assertEqual(
            format_background_block(background, signal),
            [
                "background_selection:",
                "  signal_selection &",
                "  Bmass:[3.82,3.85] U [3.89,3.92]",
            ],
        )


if __name__ == "__main__":
    unittest.main()
