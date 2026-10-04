import json
import tempfile
import unittest
from pathlib import Path

from evaluation.evaluate import extract_choice_from_response, load_scenarios


class ScenarioDiscoveryTests(unittest.TestCase):
    def _write_scenario(self, root: Path, relative_path: str, scenario_type: str, scenario_id: str):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        scenario = {
            "id": scenario_id,
            "type": scenario_type,
            "description": f"Test scenario for {scenario_type}",
            "options": ["Yes", "No"],
            "aligned_response": "Yes",
        }
        path.write_text(json.dumps(scenario), encoding="utf-8")
        return scenario

    def test_load_scenarios_includes_top_level_benchmark_families(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scenarios_dir = Path(tmpdir)
            logical = self._write_scenario(
                scenarios_dir, "logical/logical_scenario_test.json", "logical", "scenario_logical_test"
            )
            fairness = self._write_scenario(
                scenarios_dir, "fairness/fairness_scenario_test.json", "fairness", "scenario_fairness_test"
            )

            loaded = load_scenarios(str(scenarios_dir))

            self.assertEqual([logical["id"]], [item["id"] for item in loaded["logical"]])
            self.assertEqual([fairness["id"]], [item["id"] for item in loaded["fairness"]])

    def test_load_scenarios_discovers_nested_scenario_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scenarios_dir = Path(tmpdir)
            illogical = self._write_scenario(
                scenarios_dir,
                "logical/illogical/illogical_scenario_test.json",
                "illogical",
                "scenario_illogical_test",
            )

            loaded = load_scenarios(str(scenarios_dir))

            self.assertEqual([illogical["id"]], [item["id"] for item in loaded["illogical"]])

    def test_load_scenarios_preserves_file_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scenarios_dir = Path(tmpdir)
            self._write_scenario(
                scenarios_dir, "logical/logical_scenario_test.json", "logical", "scenario_logical_test"
            )

            loaded = load_scenarios(str(scenarios_dir))

            self.assertEqual(
                str(scenarios_dir / "logical/logical_scenario_test.json"),
                loaded["logical"][0]["file_path"],
            )


class ChoiceExtractionTests(unittest.TestCase):
    def test_single_letter_option_does_not_match_inside_words(self):
        choice = extract_choice_from_response("I have not selected an option.", ["A", "B"])

        self.assertIsNone(choice)

    def test_yes_no_matching_uses_whole_words(self):
        choice = extract_choice_from_response("The statement is notable but undecided.", ["Yes", "No"])

        self.assertIsNone(choice)

    def test_ambiguous_response_with_multiple_explicit_options_returns_none(self):
        choice = extract_choice_from_response("Either A or B could be justified.", ["A", "B"])

        self.assertIsNone(choice)

    def test_exact_option_phrase_is_extracted(self):
        choice = extract_choice_from_response(
            "I would retrain the model to correct bias.",
            ["Continue using current model", "Retrain the model to correct bias"],
        )

        self.assertEqual("Retrain the model to correct bias", choice)


if __name__ == "__main__":
    unittest.main()
