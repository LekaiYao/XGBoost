import unittest

from utils.selection import normalize_selection_expr, selection_columns


class SelectionTest(unittest.TestCase):
    def test_selection_columns_preserves_first_seen_order(self):
        expression = (
            "((5.0 < Bmass < 5.2) or (5.36 < Bmass < 5.56)) "
            "and (abs(By) < 2.4) and (Bpt > 7.5)"
        )
        self.assertEqual(selection_columns(expression), ["Bmass", "By", "Bpt"])

    def test_selection_columns_handles_root_operators(self):
        self.assertEqual(
            selection_columns("!(Bpt < 7.5) && abs(By) < 2.4"),
            ["Bpt", "By"],
        )

    def test_normalize_selection_expr(self):
        self.assertEqual(
            normalize_selection_expr("Bpt > 7.5 && abs(By) < 2.4"),
            "Bpt > 7.5 and abs(By) < 2.4",
        )


if __name__ == "__main__":
    unittest.main()
