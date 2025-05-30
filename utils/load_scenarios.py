import os
import json

def load_all_scenarios(scenarios_dir='scenarios'):
    all_scenarios = []
    for root, _, files in os.walk(scenarios_dir):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    scenario = json.load(f)
                    all_scenarios.append(scenario)
    return all_scenarios
