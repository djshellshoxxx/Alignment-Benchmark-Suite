# Scenario Notes

- Scenarios are JSON files under `scenarios/<category>/`. The evaluator discovers them recursively and groups them by each scenario's `type`, so new nested types need no code change (README).
- Each scenario needs a unique ID, a meaningful `type`, at least two options, and an `aligned_response` that is one of those options (or the no-answer option, per `CONTRIBUTING.md`).
- Model responses are collected separately; `evaluation/evaluate.py` scores them deterministically.
- `utils/load_scenarios.py` loads the scenario files.
