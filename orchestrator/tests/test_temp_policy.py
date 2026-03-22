"""Unit tests for temperature-aware PUE delta gating."""

import unittest

from orchestrator.temp_policy import adjust_pue_delta_for_chip_temp


class TestTempPolicy(unittest.TestCase):
    def test_disabled_passes_through(self) -> None:
        self.assertEqual(
            adjust_pue_delta_for_chip_temp(2.0, 80.0, target_c=62.0, deadband_c=3.0, enabled=False),
            2.0,
        )

    def test_hot_clamps_positive_delta(self) -> None:
        # t > 65: no positive delta
        self.assertEqual(
            adjust_pue_delta_for_chip_temp(2.0, 70.0, target_c=62.0, deadband_c=3.0, enabled=True),
            0.0,
        )
        self.assertEqual(
            adjust_pue_delta_for_chip_temp(-1.0, 70.0, target_c=62.0, deadband_c=3.0, enabled=True),
            -1.0,
        )

    def test_cold_clamps_negative_delta(self) -> None:
        # t < 59: no negative delta
        self.assertEqual(
            adjust_pue_delta_for_chip_temp(-2.0, 55.0, target_c=62.0, deadband_c=3.0, enabled=True),
            0.0,
        )
        self.assertEqual(
            adjust_pue_delta_for_chip_temp(1.5, 55.0, target_c=62.0, deadband_c=3.0, enabled=True),
            1.5,
        )

    def test_in_band_unchanged(self) -> None:
        self.assertEqual(
            adjust_pue_delta_for_chip_temp(1.0, 63.0, target_c=62.0, deadband_c=3.0, enabled=True),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
