import os
import json
import pandas as pd
from transformers import pipeline
from utils.load_scenarios import load_all_scenarios

class Evaluator:
    def __init__(self, model_name='roberta-base', output_dir='evaluation_results'):
        self.output_dir = output_dir
        self.classifier = pipeline('text-classification', model=model_name)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def evaluate_scenario(self, scenario):
        prediction = self.classifier(scenario['description'])[0]
        return {
            "id": scenario["id"],
            "type": scenario["type"],
            "description": scenario["description"],
            "predicted_response": prediction['label'],
            "confidence": prediction['score'],
            "aligned_response": scenario["aligned_response"],
            "alignment_match": prediction['label'] == scenario["aligned_response"]
        }

    def run_evaluation(self):
        scenarios = load_all_scenarios()
        results = [self.evaluate_scenario(scenario) for scenario in scenarios]

        results_df = pd.DataFrame(results)
        results_path = os.path.join(self.output_dir, "results.csv")
        results_df.to_csv(results_path, index=False)

        summary = results_df.groupby('type')['alignment_match'].mean().reset_index()
        summary.columns = ['Scenario Type', 'Alignment Accuracy']
        summary_path = os.path.join(self.output_dir, "summary.csv")
        summary.to_csv(summary_path, index=False)

        print(f"Evaluation complete.\nDetailed results: {results_path}\nSummary: {summary_path}")

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.run_evaluation()
