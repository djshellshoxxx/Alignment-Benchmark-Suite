# Developer Guide

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v     # 10 tests pass with requirements installed
```

Evaluate: `python evaluation/evaluate.py --responses_file responses.json --output_file evaluation_results.json`.
Analyse: `python evaluation/analyze_results.py --results_file evaluation_results.json --output_dir analysis_output --generate_plots`.

CI (`.github/workflows/tests.yml`) runs the unit tests. Evaluator changes need tests that cover scenario discovery and scoring (`CONTRIBUTING.md`).
