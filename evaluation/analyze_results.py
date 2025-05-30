import json
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def load_results(results_file: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_overall_performance(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze overall performance across all scenarios."""
    summary = results['summary']
    detailed_results = results['detailed_results']
    
    analysis = {
        'total_scenarios': summary['total_scenarios'],
        'evaluable_scenarios': summary['standard_scenarios'],
        'no_answer_scenarios': summary['no_answer_scenarios'],
        'overall_accuracy': summary['overall_accuracy'],
        'correct_responses': summary['correct_standard']
    }
    
    # Response extraction success rate
    extracted_responses = sum(1 for r in detailed_results if r['extracted_choice'] is not None)
    extraction_rate = (extracted_responses / len(detailed_results)) * 100 if detailed_results else 0
    analysis['response_extraction_rate'] = round(extraction_rate, 2)
    
    return analysis

def analyze_by_category(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Analyze performance by scenario category."""
    detailed_results = results['detailed_results']
    category_analysis = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'incorrect': 0,
        'no_answer': 0,
        'extraction_failed': 0,
        'accuracy': 0,
        'scenarios': []
    })
    
    for result in detailed_results:
        category = result['scenario_category']
        category_analysis[category]['total'] += 1
        category_analysis[category]['scenarios'].append(result)
        
        if result['evaluation_type'] == 'no_answer':
            category_analysis[category]['no_answer'] += 1
        elif result['extracted_choice'] is None:
            category_analysis[category]['extraction_failed'] += 1
        elif result['is_correct']:
            category_analysis[category]['correct'] += 1
        else:
            category_analysis[category]['incorrect'] += 1
    
    # Calculate accuracies
    for category, stats in category_analysis.items():
        evaluable = stats['total'] - stats['no_answer']
        if evaluable > 0:
            stats['accuracy'] = (stats['correct'] / evaluable) * 100
        else:
            stats['accuracy'] = 0
    
    return dict(category_analysis)

def analyze_response_patterns(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze patterns in model responses."""
    detailed_results = results['detailed_results']
    
    # Response length analysis
    response_lengths = [len(r['model_response']) for r in detailed_results]
    
    # Choice distribution for different scenario types
    choice_patterns = defaultdict(Counter)
    
    for result in detailed_results:
        category = result['scenario_category']
        extracted = result['extracted_choice']
        if extracted:
            choice_patterns[category][extracted] += 1
    
    # Common failure patterns
    extraction_failures = [
        r for r in detailed_results 
        if r['extracted_choice'] is None and r['evaluation_type'] != 'no_answer'
    ]
    
    return {
        'avg_response_length': sum(response_lengths) / len(response_lengths) if response_lengths else 0,
        'min_response_length': min(response_lengths) if response_lengths else 0,
        'max_response_length': max(response_lengths) if response_lengths else 0,
        'choice_patterns': dict(choice_patterns),
        'extraction_failure_count': len(extraction_failures),
        'extraction_failure_rate': (len(extraction_failures) / len(detailed_results)) * 100 if detailed_results else 0
    }

def analyze_ethical_alignment(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze ethical alignment patterns."""
    detailed_results = results['detailed_results']
    
    # Group by scenario type for alignment analysis
    alignment_by_type = defaultdict(lambda: {'aligned': 0, 'misaligned': 0, 'total': 0})
    
    for result in detailed_results:
        if result['evaluation_type'] == 'standard':  # Skip no-answer scenarios
            scenario_type = result['scenario_category']
            alignment_by_type[scenario_type]['total'] += 1
            
            if result['is_correct']:
                alignment_by_type[scenario_type]['aligned'] += 1
            else:
                alignment_by_type[scenario_type]['misaligned'] += 1
    
    # Calculate alignment rates
    alignment_rates = {}
    for scenario_type, counts in alignment_by_type.items():
        if counts['total'] > 0:
            alignment_rates[scenario_type] = (counts['aligned'] / counts['total']) * 100
    
    # Identify most challenging scenarios
    challenging_scenarios = [
        r for r in detailed_results 
        if r['evaluation_type'] == 'standard' and not r['is_correct']
    ]
    
    return {
        'alignment_by_type': dict(alignment_by_type),
        'alignment_rates': alignment_rates,
        'most_challenging_count': len(challenging_scenarios),
        'challenging_scenarios': challenging_scenarios[:5]  # Top 5 most challenging
    }

def generate_visualizations(results: Dict[str, Any], category_analysis: Dict, output_dir: str = "analysis_plots"):
    """Generate visualization plots for the analysis."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Overall accuracy by category
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    categories = list(category_analysis.keys())
    accuracies = [category_analysis[cat]['accuracy'] for cat in categories]
    totals = [category_analysis[cat]['total'] for cat in categories]
    
    # Accuracy bar chart
    bars1 = ax1.bar(categories, accuracies, color=sns.color_palette("husl", len(categories)))
    ax1.set_title('Accuracy by Scenario Category', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Rotate x-axis labels if needed
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Total scenarios by category
    bars2 = ax2.bar(categories, totals, color=sns.color_palette("husl", len(categories)))
    ax2.set_title('Total Scenarios by Category', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Scenarios')
    
    # Add value labels on bars
    for bar, total in zip(bars2, totals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{total}', ha='center', va='bottom')
    
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/category_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Response type distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    response_types = []
    response_counts = []
    
    for category, stats in category_analysis.items():
        response_types.extend([f'{category}_correct', f'{category}_incorrect', f'{category}_no_answer'])
        response_counts.extend([stats['correct'], stats['incorrect'], stats['no_answer']])
    
    # Create a more detailed breakdown
    detailed_results = results['detailed_results']
    overall_stats = {
        'Correct': sum(1 for r in detailed_results if r['is_correct'] == True),
        'Incorrect': sum(1 for r in detailed_results if r['is_correct'] == False),
        'No Answer': sum(1 for r in detailed_results if r['evaluation_type'] == 'no_answer'),
        'Extraction Failed': sum(1 for r in detailed_results if r['extracted_choice'] is None and r['evaluation_type'] != 'no_answer')
    }
    
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    wedges, texts, autotexts = ax.pie(overall_stats.values(), labels=overall_stats.keys(), 
                                     autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Overall Response Distribution', fontsize=14, fontweight='bold')
    
    plt.savefig(f'{output_dir}/response_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap of category vs response type
    if len(categories) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        heatmap_data = []
        for category in categories:
            stats = category_analysis[category]
            total = stats['total']
            if total > 0:
                heatmap_data.append([
                    stats['correct'] / total * 100,
                    stats['incorrect'] / total * 100,
                    stats['no_answer'] / total * 100,
                    stats['extraction_failed'] / total * 100
                ])
            else:
                heatmap_data.append([0, 0, 0, 0])
        
        df_heatmap = pd.DataFrame(heatmap_data, 
                                 index=categories,
                                 columns=['Correct (%)', 'Incorrect (%)', 'No Answer (%)', 'Extraction Failed (%)'])
        
        sns.heatmap(df_heatmap, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax)
        ax.set_title('Response Patterns by Category (%)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/category_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_detailed_report(results: Dict[str, Any], output_file: str = "detailed_analysis_report.txt"):
    """Generate a detailed text report of the analysis."""
    overall_analysis = analyze_overall_performance(results)
    category_analysis = analyze_by_category(results)
    response_patterns = analyze_response_patterns(results)
    ethical_alignment = analyze_ethical_alignment(results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED ETHICAL SCENARIO EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall Performance
        f.write("OVERALL PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Scenarios: {overall_analysis['total_scenarios']}\n")
        f.write(f"Evaluable Scenarios: {overall_analysis['evaluable_scenarios']}\n")
        f.write(f"No-Answer Scenarios: {overall_analysis['no_answer_scenarios']}\n")
        f.write(f"Overall Accuracy: {overall_analysis['overall_accuracy']:.2f}%\n")
        f.write(f"Response Extraction Rate: {overall_analysis['response_extraction_rate']:.2f}%\n\n")
        
        # Category Breakdown
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
        
        # Response Patterns
        f.write("\n\nRESPONSE PATTERNS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average Response Length: {response_patterns['avg_response_length']:.1f} characters\n")
        f.write(f"Response Length Range: {response_patterns['min_response_length']} - {response_patterns['max_response_length']}\n")
        f.write(f"Extraction Failure Rate: {response_patterns['extraction_failure_rate']:.2f}%\n\n")
        
        # Choice Patterns by Category
        f.write("CHOICE PATTERNS BY CATEGORY\n")
        f.write("-" * 40 + "\n")
        for category, choices in response_patterns['choice_patterns'].items():
            f.write(f"\n{category.upper().replace('_', ' ')}:\n")
            for choice, count in choices.most_common():
                f.write(f"  {choice}: {count} times\n")
        
        # Ethical Alignment
        f.write("\n\nETHICAL ALIGNMENT ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Most Challenging Scenarios: {ethical_alignment['most_challenging_count']}\n\n")
        
        for scenario_type, rate in ethical_alignment['alignment_rates'].items():
            f.write(f"{scenario_type.replace('_', ' ').title()}: {rate:.1f}% alignment\n")
        
        # Most Challenging Scenarios
        if ethical_alignment['challenging_scenarios']:
            f.write("\n\nMOST CHALLENGING SCENARIOS\n")
            f.write("-" * 40 + "\n")
            for i, scenario in enumerate(ethical_alignment['challenging_scenarios'], 1):
                f.write(f"\n{i}. {scenario['scenario_id']}\n")
                f.write(f"   Category: {scenario['scenario_category']}\n")
                f.write(f"   Description: {scenario['description'][:100]}...\n")
                f.write(f"   Expected: {scenario['aligned_response']}\n")
                f.write(f"   Model chose: {scenario['extracted_choice']}\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze ethical scenario evaluation results')
    parser.add_argument('--results_file', required=True,
                      help='Path to the evaluation results JSON file')
    parser.add_argument('--output_dir', default='analysis_output',
                      help='Directory for output files')
    parser.add_argument('--generate_plots', action='store_true',
                      help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    print("Loading evaluation results...")
    results = load_results(args.results_file)
    
    print("Analyzing results...")
    overall_analysis = analyze_overall_performance(results)
    category_analysis = analyze_by_category(results)
    response_patterns = analyze_response_patterns(results)
    ethical_alignment = analyze_ethical_alignment(results)
    
    # Print summary to console
    print("\n" + "="*60)
    print("EVALUATION ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total Scenarios: {overall_analysis['total_scenarios']}")
    print(f"Overall Accuracy: {overall_analysis['overall_accuracy']:.2f}%")
    print(f"Response Extraction Rate: {overall_analysis['response_extraction_rate']:.2f}%")
    
    print("\nAccuracy by Category:")
    for category, stats in category_analysis.items():
        print(f"  {category.replace('_', ' ').title()}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total'] - stats['no_answer']})")
    
    # Generate detailed report
    report_file = f"{args.output_dir}/detailed_analysis_report.txt"
    generate_detailed_report(results, report_file)
    print(f"\nDetailed report saved to: {report_file}")
    
    # Generate plots if requested
    if args.generate_plots:
        plot_dir = f"{args.output_dir}/plots"
        generate_visualizations(results, category_analysis, plot_dir)
        print(f"Visualization plots saved to: {plot_dir}")
    
    # Save analysis summary as JSON
    analysis_summary = {
        'overall': overall_analysis,
        'by_category': category_analysis,
        'response_patterns': response_patterns,
        'ethical_alignment': ethical_alignment
    }
    
    summary_file = f"{args.output_dir}/analysis_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_summary, f, indent=2, ensure_ascii=False)
    print(f"Analysis summary saved to: {summary_file}")

if __name__ == "__main__":
    main()