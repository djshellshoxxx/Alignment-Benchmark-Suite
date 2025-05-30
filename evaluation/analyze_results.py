# analyze_results.py
import pandas as pd

def analyze_results(results_file='evaluation_results/results.csv'):
    df = pd.read_csv(results_file)

    alignment_accuracy = df['alignment_match'].mean() * 100
    print(f"Overall alignment accuracy: {alignment_accuracy:.2f}%")

    accuracy_by_type = df.groupby('type')['alignment_match'].mean() * 100
    print("\nAccuracy by scenario type:")
    print(accuracy_by_type)

if __name__ == "__main__":
    analyze_results()
