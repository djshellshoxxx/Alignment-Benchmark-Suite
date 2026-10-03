import json
import tempfile
import unittest
from pathlib import Path

from evaluation.evaluate import load_scenarios


class ScenarioDiscoveryTests(unittest.TestCase):
    def test_load_scenarios_includes_logical_scenarios(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scenarios_dir = Path(tmpdir)
            logical_dir = scenarios_dir / "logical"
            logical_dir.mkdir(parents=True)
            scenario = {
                "id": "scenario_logical_test",
                "type": "logical",
                "description": "If A implies B and A is true, is B true?",
                "options": ["Yes", "No"],
                "aligned_response": "Yes",
            }
            scenario_path = logical_dir / "logical_scenario_test.json"
            scenario_path.write_text(json.dumps(scenario), encoding="utf-8")

            loaded = load_scenarios(str(scenarios_dir))

            self.assertIn("logical", loaded)
            self.assertEqual([scenario["id"]], [item["id"] for item in loaded["logical"]])


if __name__ == "__main__":
    unittest.main()
