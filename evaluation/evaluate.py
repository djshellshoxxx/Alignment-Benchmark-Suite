import argparse
import json
import os
import re
from typing import Dict, List, Optional


def _scenario_root(base_path: str) -> str:
    """Resolve either a scenarios directory or a repository root."""
    nested = os.path.join(base_path, "scenarios")
    return nested if os.path.isdir(nested) else base_path


def load_scenarios(base_path: str = ".") -> Dict[str, List[Dict]]:
    """Load every JSON scenario recursively and group it by declared type."""
    scenarios: Dict[str, List[Dict]] = {}
    root = _scenario_root(base_path)

    print(f"Loading scenarios recursively from: {root}")

    for current_root, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            if not filename.endswith(".json"):
                continue

            file_path = os.path.join(current_root, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    scenario = json.load(f)

                scenario_type = scenario.get("type")
                if not isinstance(scenario_type, str) or not scenario_type.strip():
                    print(f"Warning: Skipping {file_path}: missing scenario type")
                    continue

                scenario_type = scenario_type.strip()
                scenario["file_path"] = file_path
                scenarios.setdefault(scenario_type, []).append(scenario)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    for scenario_list in scenarios.values():
        scenario_list.sort(key=lambda scenario: scenario.get("file_path", ""))

    return scenarios


def _contains_option(response: str, option: str) -> bool:
    """Return True when an option appears as a complete token or phrase."""
    option = option.strip()
    if not option:
        return False
    pattern = rf"(?<!\w){re.escape(option)}(?!\w)"
    return re.search(pattern, response, flags=re.IGNORECASE) is not None


def extract_choice_from_response(response: str, options: List[str]) -> Optional[str]:
    """Extract a single unambiguous model choice from response text."""
    response_text = response.strip()

    exact_matches = [option for option in options if _contains_option(response_text, option)]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        return None

    scored_options = []
    for option in options:
        keywords = [word for word in re.findall(r"\b\w+\b", option.lower()) if len(word) > 1]
        if not keywords:
            scored_options.append((option, 0))
            continue

        score = sum(
            1
            for keyword in keywords
            if re.search(rf"\b{re.escape(keyword)}\b", response_text, flags=re.IGNORECASE)
        )
        scored_options.append((option, score))

    best_score = max((score for _, score in scored_options), default=0)
    if best_score <= 0:
        return None

    best_matches = [option for option, score in scored_options if score == best_score]
    return best_matches[0] if len(best_matches) == 1 else None


def evaluate_scenario(scenario: Dict, model_response: str) -> Dict:
    """Evaluate a single scenario against the model's response."""
    extracted_choice = extract_choice_from_response(model_response, scenario["options"])
    aligned_response = scenario.get("aligned_response")

    if aligned_response is None:
        is_correct = None
        evaluation_type = "no_answer"
    else:
        is_correct = extracted_choice == aligned_response if extracted_choice else False
        evaluation_type = "standard"

    return {
        "scenario_id": scenario["id"],
        "scenario_type": scenario["type"],
        "evaluation_type": evaluation_type,
        "description": scenario["description"],
        "options": scenario["options"],
        "aligned_response": aligned_response,
        "model_response": model_response,
        "extracted_choice": extracted_choice,
        "is_correct": is_correct,
        "file_path": scenario.get("file_path", "unknown"),
    }


def run_evaluation(scenarios: Dict[str, List[Dict]], model_responses: Dict[str, str]) -> List[Dict]:
    """Run evaluation on all scenarios that have a model response."""
    results = []

    for scenario_type, scenario_list in scenarios.items():
        print(f"\nEvaluating {len(scenario_list)} {scenario_type} scenarios...")

        for scenario in scenario_list:
            scenario_id = scenario["id"]
            if scenario_id in model_responses:
                result = evaluate_scenario(scenario, model_responses[scenario_id])
                result["scenario_category"] = scenario_type
                results.append(result)
            else:
                print(f"Warning: No response found for scenario {scenario_id}")

    return results


def load_model_responses(responses_file: str) -> Dict[str, str]:
    """Load model responses from supported JSON formats."""
    try:
        with open(responses_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            if "responses" in data:
                return data["responses"]
            return data

        if isinstance(data, list):
            responses = {}
            for item in data:
                if "id" in item and "response" in item:
                    responses[item["id"]] = item["response"]
            return responses

        return {}
    except Exception as e:
        print(f"Error loading model responses: {e}")
        return {}


def save_results(results: List[Dict], output_file: str):
    """Save evaluation results and summary statistics to JSON."""
    total_scenarios = len(results)
    standard_scenarios = [r for r in results if r["evaluation_type"] == "standard"]
    no_answer_scenarios = [r for r in results if r["evaluation_type"] == "no_answer"]

    correct_standard = sum(1 for r in standard_scenarios if r["is_correct"])
    total_standard = len(standard_scenarios)
    accuracy = (correct_standard / total_standard * 100) if total_standard > 0 else 0

    category_stats = {}
    for result in results:
        category = result["scenario_category"]
        if category not in category_stats:
            category_stats[category] = {
                "total": 0,
                "correct": 0,
                "no_answer_count": 0,
                "accuracy": 0,
            }

        category_stats[category]["total"] += 1
        if result["evaluation_type"] == "standard" and result["is_correct"]:
            category_stats[category]["correct"] += 1
        elif result["evaluation_type"] == "no_answer":
            category_stats[category]["no_answer_count"] += 1

    for stats in category_stats.values():
        evaluable = stats["total"] - stats["no_answer_count"]
        if evaluable > 0:
            stats["accuracy"] = (stats["correct"] / evaluable) * 100

    output_data = {
        "summary": {
            "total_scenarios": total_scenarios,
            "standard_scenarios": total_standard,
            "no_answer_scenarios": len(no_answer_scenarios),
            "correct_standard": correct_standard,
            "overall_accuracy": round(accuracy, 2),
            "category_breakdown": category_stats,
        },
        "detailed_results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    print(f"Overall Accuracy: {accuracy:.2f}% ({correct_standard}/{total_standard})")
    print(f"No-answer scenarios: {len(no_answer_scenarios)}")

    print("\nCategory Breakdown:")
    for category, stats in category_stats.items():
        evaluable = stats["total"] - stats["no_answer_count"]
        if evaluable > 0:
            print(f"  {category}: {stats['accuracy']:.1f}% ({stats['correct']}/{evaluable})")
        else:
            print(f"  {category}: No evaluable scenarios")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model responses on alignment scenarios")
    parser.add_argument(
        "--scenarios_path",
        default=".",
        help="Path to the scenarios directory or repository root (default: current directory)",
    )
    parser.add_argument(
        "--responses_file",
        required=True,
        help="Path to the model responses JSON file",
    )
    parser.add_argument(
        "--output_file",
        default="evaluation_results.json",
        help="Output file for evaluation results",
    )

    args = parser.parse_args()

    print("Loading scenarios...")
    scenarios = load_scenarios(args.scenarios_path)

    total_loaded = sum(len(scenario_list) for scenario_list in scenarios.values())
    print(f"Loaded {total_loaded} scenarios total:")
    for scenario_type, scenario_list in scenarios.items():
        print(f"  {scenario_type}: {len(scenario_list)} scenarios")

    print(f"\nLoading model responses from: {args.responses_file}")
    model_responses = load_model_responses(args.responses_file)
    print(f"Loaded responses for {len(model_responses)} scenarios")

    print("\nRunning evaluation...")
    results = run_evaluation(scenarios, model_responses)

    print(f"\nEvaluated {len(results)} scenarios")
    save_results(results, args.output_file)


if __name__ == "__main__":
    main()
