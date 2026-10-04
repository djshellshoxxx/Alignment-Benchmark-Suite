import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(results_file: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_overall_performance(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze overall performance across all scenarios."""
    summary = results["summary"]
    detailed_results = results["detailed_results"]

    analysis = {
        "total_scenarios": summary["total_scenarios"],
        "evaluable_scenarios": summary["standard_scenarios"],
        "no_answer_scenarios": summary["no_answer_scenarios"],
        "overall_accuracy": summary["overall_accuracy"],
        "correct_responses": summary["correct_standard"],
    }

    extracted_responses = sum(1 for result in detailed_results if result["extracted_choice"] is not None)
    extraction_rate = (extracted_responses / len(detailed_results)) * 100 if detailed_results else 0
    analysis["response_extraction_rate"] = round(extraction_rate, 2)

    return analysis


def analyze_by_category(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Analyze performance by scenario category."""
    detailed_results = results["detailed_results"]
    category_analysis = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "no_answer": 0,
            "extraction_failed": 0,
            "accuracy": 0,
            "scenarios": [],
        }
    )

    for result in detailed_results:
        category = result["scenario_category"]
        stats = category_analysis[category]
        stats["total"] += 1
        stats["scenarios"].append(result)

        if result["evaluation_type"] == "no_answer":
            stats["no_answer"] += 1
        elif result["extracted_choice"] is None:
            stats["extraction_failed"] += 1
        elif result["is_correct"]:
            stats["correct"] += 1
        else:
            stats["incorrect"] += 1

    for stats in category_analysis.values():
        evaluable = stats["total"] - stats["no_answer"]
        stats["accuracy"] = (stats["correct"] / evaluable) * 100 if evaluable > 0 else 0

    return dict(category_analysis)


def analyze_response_patterns(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze patterns in model responses."""
    detailed_results = results["detailed_results"]
    response_lengths = [len(result["model_response"]) for result in detailed_results]
    choice_patterns = defaultdict(Counter)

    for result in detailed_results:
        extracted = result["extracted_choice"]
        if extracted:
            choice_patterns[result["scenario_category"]][extracted] += 1

    extraction_failures = [
        result
        for result in detailed_results
        if result["extracted_choice"] is None and result["evaluation_type"] != "no_answer"
    ]

    return {
        "avg_response_length": sum(response_lengths) / len(response_lengths) if response_lengths else 0,
        "min_response_length": min(response_lengths) if response_lengths else 0,
        "max_response_length": max(response_lengths) if response_lengths else 0,
        "choice_patterns": dict(choice_patterns),
        "extraction_failure_count": len(extraction_failures),
        "extraction_failure_rate": (
            len(extraction_failures) / len(detailed_results) * 100 if detailed_results else 0
        ),
    }


def analyze_ethical_alignment(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze alignment patterns for all evaluable scenario types."""
    detailed_results = results["detailed_results"]
    alignment_by_type = defaultdict(lambda: {"aligned": 0, "misaligned": 0, "total": 0})

    for result in detailed_results:
        if result["evaluation_type"] != "standard":
            continue

        scenario_type = result["scenario_category"]
        alignment_by_type[scenario_type]["total"] += 1
        if result["is_correct"]:
            alignment_by_type[scenario_type]["aligned"] += 1
        else:
            alignment_by_type[scenario_type]["misaligned"] += 1

    alignment_rates = {}
    for scenario_type, counts in alignment_by_type.items():
        if counts["total"] > 0:
            alignment_rates[scenario_type] = counts["aligned"] / counts["total"] * 100

    challenging_scenarios = [
        result
        for result in detailed_results
        if result["evaluation_type"] == "standard" and not result["is_correct"]
    ]

    return {
        "alignment_by_type": dict(alignment_by_type),
        "alignment_rates": alignment_rates,
        "most_challenging_count": len(challenging_scenarios),
        "challenging_scenarios": challenging_scenarios[:5],
    }


def build_response_distribution_stats(results: Dict[str, Any]) -> Dict[str, int]:
    """Partition evaluated responses into mutually exclusive outcome buckets."""
    stats = {
        "Correct": 0,
        "Incorrect": 0,
        "No Answer": 0,
        "Extraction Failed": 0,
    }

    for result in results.get("detailed_results", []):
        if result.get("evaluation_type") == "no_answer":
            stats["No Answer"] += 1
        elif result.get("extracted_choice") is None:
            stats["Extraction Failed"] += 1
        elif result.get("is_correct") is True:
            stats["Correct"] += 1
        else:
            stats["Incorrect"] += 1

    return stats


def generate_visualizations(
    results: Dict[str, Any], category_analysis: Dict[str, Any], output_dir: str = "analysis_plots"
):
    """Generate visualization plots for the analysis."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.style.use("default")
    sns.set_palette("husl")

    categories = list(category_analysis.keys())
    accuracies = [category_analysis[category]["accuracy"] for category in categories]
    totals = [category_analysis[category]["total"] for category in categories]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    if categories:
        palette = sns.color_palette("husl", len(categories))
        bars1 = ax1.bar(categories, accuracies, color=palette)
        for bar, accuracy in zip(bars1, accuracies):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{accuracy:.1f}%",
                ha="center",
                va="bottom",
            )

        bars2 = ax2.bar(categories, totals, color=palette)
        for bar, total in zip(bars2, totals):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(total),
                ha="center",
                va="bottom",
            )

        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    else:
        ax1.text(0.5, 0.5, "No category results", ha="center", va="center", transform=ax1.transAxes)
        ax2.text(0.5, 0.5, "No category results", ha="center", va="center", transform=ax2.transAxes)

    ax1.set_title("Accuracy by Scenario Category", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax2.set_title("Total Scenarios by Category", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Number of Scenarios")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    overall_stats = build_response_distribution_stats(results)
    if sum(overall_stats.values()) > 0:
        colors = ["#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
        ax.pie(
            overall_stats.values(),
            labels=overall_stats.keys(),
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
    else:
        ax.text(0.5, 0.5, "No evaluated responses", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    ax.set_title("Overall Response Distribution", fontsize=14, fontweight="bold")

    plt.savefig(f"{output_dir}/response_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    if len(categories) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        heatmap_data = []
        for category in categories:
            stats = category_analysis[category]
            total = stats["total"]
            if total > 0:
                heatmap_data.append(
                    [
                        stats["correct"] / total * 100,
                        stats["incorrect"] / total * 100,
                        stats["no_answer"] / total * 100,
                        stats["extraction_failed"] / total * 100,
                    ]
                )
            else:
                heatmap_data.append([0, 0, 0, 0])

        df_heatmap = pd.DataFrame(
            heatmap_data,
            index=categories,
            columns=["Correct (%)", "Incorrect (%)", "No Answer (%)", "Extraction Failed (%)"],
        )
        sns.heatmap(df_heatmap, annot=True, fmt=".1f", cmap="RdYlGn_r", ax=ax)
        ax.set_title("Response Patterns by Category (%)", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/category_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()


def generate_detailed_report(results: Dict[str, Any], output_file: str = "detailed_analysis_report.txt"):
    """Generate a detailed text report of the analysis."""
    overall_analysis = analyze_overall_performance(results)
    category_analysis = analyze_by_category(results)
    response_patterns = analyze_response_patterns(results)
    ethical_alignment = analyze_ethical_alignment(results)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED ETHICAL SCENARIO EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("OVERALL PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Scenarios: {overall_analysis['total_scenarios']}\n")
        f.write(f"Evaluable Scenarios: {overall_analysis['evaluable_scenarios']}\n")
        f.write(f"No-Answer Scenarios: {overall_analysis['no_answer_scenarios']}\n")
        f.write(f"Overall Accuracy: {overall_analysis['overall_accuracy']:.2f}%\n")
        f.write(f"Response Extraction Rate: {overall_analysis['response_extraction_rate']:.2f}%\n\n")

        f.write("PERFORMANCE BY CATEGORY\n")
        f.write("-" * 40 + "\n")
        for category, stats in category_analysis.items():
            f.write(f"\n{category.upper().replace('_', ' ')}:\n")
            f.write(f"  Total Scenarios: {stats['total']}\n")
            f.write(f"  Correct: {stats['correct']}\n")
            f.write(f"  Incorrect: {stats['incorrect']}\n")
            f.write(f"  No Answer: {stats['no_answer']}\n")
            f.write(f"  Extraction Failed: {stats['extraction_failed']}\n")
            f.write(f"  Accuracy: {stats['accuracy']:.2f}%\n")

        f.write("\n\nRESPONSE PATTERNS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average Response Length: {response_patterns['avg_response_length']:.1f} characters\n")
        f.write(
            f"Response Length Range: {response_patterns['min_response_length']} - "
            f"{response_patterns['max_response_length']}\n"
        )
        f.write(f"Extraction Failure Rate: {response_patterns['extraction_failure_rate']:.2f}%\n\n")

        f.write("CHOICE PATTERNS BY CATEGORY\n")
        f.write("-" * 40 + "\n")
        for category, choices in response_patterns["choice_patterns"].items():
            f.write(f"\n{category.upper().replace('_', ' ')}:\n")
            for choice, count in choices.most_common():
                f.write(f"  {choice}: {count} times\n")

        f.write("\n\nETHICAL ALIGNMENT ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Most Challenging Scenarios: {ethical_alignment['most_challenging_count']}\n\n")
        for scenario_type, rate in ethical_alignment["alignment_rates"].items():
            f.write(f"{scenario_type.replace('_', ' ').title()}: {rate:.1f}% alignment\n")

        if ethical_alignment["challenging_scenarios"]:
            f.write("\n\nMOST CHALLENGING SCENARIOS\n")
            f.write("-" * 40 + "\n")
            for index, scenario in enumerate(ethical_alignment["challenging_scenarios"], 1):
                f.write(f"\n{index}. {scenario['scenario_id']}\n")
                f.write(f"   Category: {scenario['scenario_category']}\n")
                f.write(f"   Description: {scenario['description'][:100]}...\n")
                f.write(f"   Expected: {scenario['aligned_response']}\n")
                f.write(f"   Model chose: {scenario['extracted_choice']}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze alignment scenario evaluation results")
    parser.add_argument("--results_file", required=True, help="Path to the evaluation results JSON file")
    parser.add_argument("--output_dir", default="analysis_output", help="Directory for output files")
    parser.add_argument("--generate_plots", action="store_true", help="Generate visualization plots")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading evaluation results...")
    results = load_results(args.results_file)

    print("Analyzing results...")
    overall_analysis = analyze_overall_performance(results)
    category_analysis = analyze_by_category(results)
    response_patterns = analyze_response_patterns(results)
    ethical_alignment = analyze_ethical_alignment(results)

    print("\n" + "=" * 60)
    print("EVALUATION ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total Scenarios: {overall_analysis['total_scenarios']}")
    print(f"Overall Accuracy: {overall_analysis['overall_accuracy']:.2f}%")
    print(f"Response Extraction Rate: {overall_analysis['response_extraction_rate']:.2f}%")

    print("\nAccuracy by Category:")
    for category, stats in category_analysis.items():
        evaluable = stats["total"] - stats["no_answer"]
        print(f"  {category.replace('_', ' ').title()}: {stats['accuracy']:.1f}% ({stats['correct']}/{evaluable})")

    report_file = f"{args.output_dir}/detailed_analysis_report.txt"
    generate_detailed_report(results, report_file)
    print(f"\nDetailed report saved to: {report_file}")

    if args.generate_plots:
        plot_dir = f"{args.output_dir}/plots"
        generate_visualizations(results, category_analysis, plot_dir)
        print(f"Visualization plots saved to: {plot_dir}")

    analysis_summary = {
        "overall": overall_analysis,
        "by_category": category_analysis,
        "response_patterns": response_patterns,
        "ethical_alignment": ethical_alignment,
    }

    summary_file = f"{args.output_dir}/analysis_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, indent=2, ensure_ascii=False)
    print(f"Analysis summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
