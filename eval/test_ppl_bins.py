import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval.ppl_bins import (
    analyze,
    frequency_rank_order,
    level_set_bins,
    load_run,
    mass_bins,
    merge_run_shards,
)


class PPLBinsTest(unittest.TestCase):
    def test_frequency_order_uses_token_id_for_ties(self):
        counts = np.array([3, 9, 3, 9, 0], dtype=np.int64)
        np.testing.assert_array_equal(
            frequency_rank_order(counts), np.array([1, 3, 0, 2, 4])
        )

    def test_mass_bins_include_threshold_crossing_token(self):
        counts = np.array([50, 30, 10, 10], dtype=np.int64)
        order = frequency_rank_order(counts)
        bins, labels = mass_bins(order, counts, [0.6, 0.9])
        # Token 1 raises cumulative mass from 50% to 80%, so it belongs to
        # the smallest prefix covering 60%, not to the next bin.
        np.testing.assert_array_equal(bins, np.array([0, 0, 1, 2]))
        self.assertEqual(labels, ["mass<=0.6", "mass(0.6,0.9]", "mass>0.9"])

    def test_repeated_run_names_merge_only_disjoint_languages(self):
        first = {"en": (np.ones(3), np.ones(3, dtype=np.int64))}
        second = {"de": (np.ones(3), np.ones(3, dtype=np.int64))}
        merged = merge_run_shards("model", [("a", first), ("b", second)])
        self.assertEqual(set(merged), {"de", "en"})
        with self.assertRaisesRegex(ValueError, "duplicate languages"):
            merge_run_shards("model", [("a", first), ("c", first)])

    def test_loader_rejects_noninteger_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez(path, en_nll=np.ones(3), en_cnt=np.ones(3, dtype=np.float64))
            with self.assertRaisesRegex(ValueError, "integer dtype"):
                load_run(path)

    def test_analysis_decomposes_gap_and_checks_exact_counts(self):
        frequency = np.array([50, 30, 10, 10], dtype=np.float64)
        order = frequency_rank_order(frequency)
        bin_ids, labels = level_set_bins(order, [2, 2])
        count_en = np.array([5, 3, 1, 1], dtype=np.int64)
        count_de = np.array([2, 2, 1, 1], dtype=np.int64)
        reference = {
            "en": (count_en.astype(float), count_en),
            "de": (count_de.astype(float), count_de),
        }
        candidate = {
            "en": (count_en.astype(float) + 0.5, count_en.copy()),
            "de": (count_de.astype(float) + 0.25, count_de.copy()),
        }
        report = analyze(
            {"reference": reference, "candidate": candidate},
            bin_ids,
            labels,
            "reference",
            frequency,
        )
        self.assertEqual(report["type_count"], [2, 2])
        self.assertAlmostEqual(sum(report["training_mass_share"]), 1.0)
        gap = report["gap_vs_reference"]["candidate"]
        self.assertAlmostEqual(
            sum(gap["per_bin_contribution"]), gap["total_mean_log_ppl_gap"]
        )

        mismatched = {
            lang: (nll.copy(), count.copy())
            for lang, (nll, count) in candidate.items()
        }
        # Preserve the first bin's pooled count while changing token-level
        # windows; the old per-bin-only validation would miss this.
        mismatched["en"][1][0] -= 1
        mismatched["en"][1][1] += 1
        with self.assertRaisesRegex(ValueError, "per-token counts differ"):
            analyze(
                {"reference": reference, "candidate": mismatched},
                bin_ids,
                labels,
                "reference",
                frequency,
            )


if __name__ == "__main__":
    unittest.main()
