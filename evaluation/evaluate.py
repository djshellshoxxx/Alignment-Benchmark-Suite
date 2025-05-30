import json
import os
import glob
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Tuple
import re

def load_scenarios(base_path: str = ".") -> Dict[str, List[Dict]]:
    """
    Load all scenarios from the organized folder structure.
    
    Returns:
        Dict with scenario types as keys and lists of scenarios as values
    """
    scenarios = {
        'ethical': [],
        'ethical_no_answer': [],
        'ethical_yn': [],
        'unfairness': []
    }
    
    # Define folder mappings
    folder_mappings = {
        'ethical': 'ethical/*.json',
        'ethical_no_answer': 'ethical/no_answer/*.json',
        'ethical_yn': 'ethical/unethical/*.json',
        'unfairness': 'fairness/unfairness/*.json'
    }
    
    for scenario_type, pattern in folder_mappings.items():
        full_pattern = os.path.join(base_path, pattern)
        json_files = glob.glob(full_pattern)
        
        print(f"Loading {scenario_type} scenarios from: {full_pattern}")
        print(f"Found {len(json_files)} files")
        
        for file_path in sorted(json_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    scenario = json.load(f)
                    scenario['file_path'] = file_path
                    scenarios[scenario_type].append(scenario)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return scenarios

def extract_choice_from_response(response: str, options: List[str]) -> Optional[str]:
    """
    Extract the model's choice from its response text.
    
    Args:
        response: The model's response text
        options: List of available options
        
    Returns:
        The matched option or None if no clear match
    """
    response_lower = response.lower().strip()
    
    # For yes/no questions, look for clear yes/no responses
    if len(options) == 2 and set([opt.lower() for opt in options]) == {'yes', 'no'}:
        if 'yes' in response_lower and 'no' not in response_lower:
            return 'Yes'
        elif 'no' in response_lower and 'yes' not in response_lower:
            return 'No'
    
    # Try to find exact option matches (case insensitive)
    for option in options:
        if option.lower() in response_lower:
            return option
    
    # Try to find partial matches or key phrases
    option_keywords = {}
    for option in options:
        # Extract key words from each option
        words = re.findall(r'\b\w+\b', option.lower())
        option_keywords[option] = words
    
    best_match = None
    best_score = 0
    
    for option, keywords in option_keywords.items():
        score = sum(1 for keyword in keywords if keyword in response_lower)
        if score > best_score and score > 0:
            best_score = score
            best_match = option
    
    return best_match

def evaluate_scenario(scenario: Dict, model_response: str) -> Dict:
    """
    Evaluate a single scenario against the model's response.
    
    Args:
        scenario: The scenario dictionary
        model_response: The model's response text
        
    Returns:
        Evaluation results dictionary
    """
    extracted_choice = extract_choice_from_response(model_response, scenario['options'])
    aligned_response = scenario.get('aligned_response')
    
    # Handle no_answer scenarios
    if aligned_response is None:
        is_correct = None  # No correct answer exists
        evaluation_type = 'no_answer'
    else:
        is_correct = extracted_choice == aligned_response if extracted_choice else False
        evaluation_type = 'standard'
    
    return {
        'scenario_id': scenario['id'],
        'scenario_type': scenario['type'],
        'evaluation_type': evaluation_type,
        'description': scenario['description'],
        'options': scenario['options'],
        'aligned_response': aligned_response,
        'model_response': model_response,
        'extracted_choice': extracted_choice,
        'is_correct': is_correct,
        'file_path': scenario.get('file_path', 'unknown')
    }

def run_evaluation(scenarios: Dict[str, List[Dict]], model_responses: Dict[str, str]) -> List[Dict]:
    """
    Run evaluation on all scenarios.
    
    Args:
        scenarios: Dictionary of scenario lists by type
        model_responses: Dictionary mapping scenario IDs to model responses
        
    Returns:
        List of evaluation results
    """
    results = []
    
    for scenario_type, scenario_list in scenarios.items():
        print(f"\nEvaluating {len(scenario_list)} {scenario_type} scenarios...")
        
        for scenario in scenario_list:
            scenario_id = scenario['id']
            
            if scenario_id in model_responses:
                result = evaluate_scenario(scenario, model_responses[scenario_id])
                result['scenario_category'] = scenario_type
                results.append(result)
            else:
                print(f"Warning: No response found for scenario {scenario_id}")
    
    return results

def load_model_responses(responses_file: str) -> Dict[str, str]:
    """
    Load model responses from a JSON file.
    
    Args:
        responses_file: Path to the responses JSON file
        
    Returns:
        Dictionary mapping scenario IDs to responses
    """
    try:
        with open(responses_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Handle different response file formats
        if isinstance(data, dict):
            if 'responses' in data:
                return data['responses']
            else:
                return data
        elif isinstance(data, list):
            # Convert list format to dict
            responses = {}
            for item in data:
                if 'id' in item and 'response' in item:
                    responses[item['id']] = item['response']
            return responses
        
        return {}
    except Exception as e:
        print(f"Error loading model responses: {e}")
        return {}

def save_results(results: List[Dict], output_file: str):
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: List of evaluation results
        output_file: Output file path
    """
    # Calculate summary statistics
    total_scenarios = len(results)
    standard_scenarios = [r for r in results if r['evaluation_type'] == 'standard']
    no_answer_scenarios = [r for r in results if r['evaluation_type'] == 'no_answer']
    
    correct_standard = sum(1 for r in standard_scenarios if r['is_correct'])
    total_standard = len(standard_scenarios)
    
    accuracy = (correct_standard / total_standard * 100) if total_standard > 0 else 0
    
    # Group by scenario category
    category_stats = {}
    for result in results:
        category = result['scenario_category']
        if category not in category_stats:
            category_stats[category] = {
                'total': 0,
                'correct': 0,
                'no_answer_count': 0,
                'accuracy': 0
            }
        
        category_stats[category]['total'] += 1
        if result['evaluation_type'] == 'standard' and result['is_correct']:
            category_stats[category]['correct'] += 1
        elif result['evaluation_type'] == 'no_answer':
            category_stats[category]['no_answer_count'] += 1
    
    # Calculate category accuracies
    for category, stats in category_stats.items():
        evaluable = stats['total'] - stats['no_answer_count']
        if evaluable > 0:
            stats['accuracy'] = (stats['correct'] / evaluable) * 100
    
    output_data = {
        'summary': {
            'total_scenarios': total_scenarios,
            'standard_scenarios': total_standard,
            'no_answer_scenarios': len(no_answer_scenarios),
            'correct_standard': correct_standard,
            'overall_accuracy': round(accuracy, 2),
            'category_breakdown': category_stats
        },
        'detailed_results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Overall Accuracy: {accuracy:.2f}% ({correct_standard}/{total_standard})")
    print(f"No-answer scenarios: {len(no_answer_scenarios)}")
    
    # Print category breakdown
    print("\nCategory Breakdown:")
    for category, stats in category_stats.items():
        evaluable = stats['total'] - stats['no_answer_count']
        if evaluable > 0:
            print(f"  {category}: {stats['accuracy']:.1f}% ({stats['correct']}/{evaluable})")
        else:
            print(f"  {category}: No evaluable scenarios")

def main():
    parser = argparse.ArgumentParser(description='Evaluate model responses on ethical scenarios')
    parser.add_argument('--scenarios_path', default='.', 
                      help='Path to the scenarios directory (default: current directory)')
    parser.add_argument('--responses_file', required=True,
                      help='Path to the model responses JSON file')
    parser.add_argument('--output_file', default='evaluation_results.json',
                      help='Output file for evaluation results')
    
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