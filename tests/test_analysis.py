import unittest

from evaluation.analyze_results import build_response_distribution_stats


class ResponseDistributionTests(unittest.TestCase):
    def test_extraction_failures_are_not_double_counted_as_incorrect(self):
        results = {
            "detailed_results": [
                {"is_correct": True, "evaluation_type": "standard", "extracted_choice": "Yes"},
                {"is_correct": False, "evaluation_type": "standard", "extracted_choice": "No"},
                {"is_correct": False, "evaluation_type": "standard", "extracted_choice": None},
                {"is_correct": None, "evaluation_type": "no_answer", "extracted_choice": None},
            ]
        }

        stats = build_response_distribution_stats(results)

        self.assertEqual(
            {
                "Correct": 1,
                "Incorrect": 1,
                "No Answer": 1,
                "Extraction Failed": 1,
            },
            stats,
        )
        self.assertEqual(len(results["detailed_results"]), sum(stats.values()))

    def test_empty_results_produce_zero_counts(self):
        stats = build_response_distribution_stats({"detailed_results": []})

        self.assertEqual(
            {
                "Correct": 0,
                "Incorrect": 0,
                "No Answer": 0,
                "Extraction Failed": 0,
            },
            stats,
        )


if __name__ == "__main__":
    unittest.main()
