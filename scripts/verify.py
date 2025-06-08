#!/usr/bin/env python3
import os
import json
import pandas as pd
import argparse
from typing import Dict

def compile_results_from_csv(csv_path: str, run_folder: str) -> Dict:
    """Compile results directly from detailed.csv"""
    df = pd.read_csv(csv_path)
    
    total = len(df)
    refusals = df['refused'].sum()
    
    # Get valid (non-refused, non-null score) results
    valid_df = df[~df['refused'] & df['score'].notna()]
    valid_tasks = len(valid_df)
    
    if valid_tasks == 0:
        print("Warning: No valid tasks found")
        return create_empty_results(run_folder)
    
    total_score = valid_df['score'].sum()
    correct_tasks = valid_df['correct'].sum()
    
    # Per-type calculations
    location_df = valid_df[valid_df['task_type'] == 'location']
    identification_df = valid_df[valid_df['task_type'] == 'identification']
    temporal_df = valid_df[valid_df['task_type'] == 'temporal']
    analysis_df = valid_df[valid_df['task_type'] == 'analysis']
    
    location_score = location_df['score'].sum()
    location_tasks = len(location_df)
    
    identification_score = identification_df['score'].sum()
    identification_tasks = len(identification_df)
    
    temporal_score = temporal_df['score'].sum()
    temporal_tasks = len(temporal_df)
    
    analysis_score = analysis_df['score'].sum()
    analysis_tasks = len(analysis_df)
    
    # Extract model and test info from run folder name
    folder_name = os.path.basename(run_folder.rstrip('/'))
    parts = folder_name.split('_')
    model_name = parts[0] if parts else "unknown"
    test_name = parts[1] if len(parts) > 1 else "unknown"
    
    return {
        "model": model_name,
        "test": test_name,
        "n": df['case_id'].nunique(),
        "total_tasks": total,
        "refusal_rate": refusals / total if total > 0 else 0,
        "location_accuracy": location_score / location_tasks if location_tasks > 0 else 0,
        "identification_accuracy": identification_score / identification_tasks if identification_tasks > 0 else 0,
        "temporal_accuracy": temporal_score / temporal_tasks if temporal_tasks > 0 else 0,
        "analysis_accuracy": analysis_score / analysis_tasks if analysis_tasks > 0 else 0,
        "overall_accuracy": total_score / valid_tasks if valid_tasks > 0 else 0,
        "task_accuracy": correct_tasks / valid_tasks if valid_tasks > 0 else 0,
    }

def create_empty_results(run_folder: str) -> Dict:
    """Create empty results structure when no valid data found"""
    folder_name = os.path.basename(run_folder.rstrip('/'))
    parts = folder_name.split('_')
    model_name = parts[0] if parts else "unknown"
    test_name = parts[1] if len(parts) > 1 else "unknown"
    
    return {
        "model": model_name,
        "test": test_name,
        "n": 0,
        "total_tasks": 0,
        "refusal_rate": 0.0,
        "location_accuracy": 0.0,
        "identification_accuracy": 0.0,
        "temporal_accuracy": 0.0,
        "analysis_accuracy": 0.0,
        "overall_accuracy": 0.0,
        "task_accuracy": 0.0,
    }

def main():
    parser = argparse.ArgumentParser(description="Reverify benchmark results from detailed.csv")
    parser.add_argument("run_folder", help="Path to the run folder containing results/detailed.csv")
    args = parser.parse_args()
    
    run_folder = args.run_folder.rstrip('/')
    csv_path = os.path.join(run_folder, "results", "detailed.csv")
    summary_path = os.path.join(run_folder, "results", "summary.json")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return 1
    
    print(f"Reverifying results from {csv_path}")
    
    try:
        results = compile_results_from_csv(csv_path, run_folder)
        
        # Save updated summary
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Updated summary saved to {summary_path}")
        print("\nResults:")
        print(f"Total samples: {results['n']}")
        print(f"Total tasks: {results['total_tasks']}")
        print(f"Refusal rate: {results['refusal_rate']:.2%}")
        print(f"Location accuracy: {results['location_accuracy']:.3f}")
        print(f"Identification accuracy: {results['identification_accuracy']:.3f}")
        print(f"Temporal accuracy: {results['temporal_accuracy']:.3f}")
        print(f"Analysis accuracy: {results['analysis_accuracy']:.3f}")
        print(f"Overall accuracy: {results['overall_accuracy']:.3f}")
        print(f"Task accuracy: {results['task_accuracy']:.3f}")
        
        return 0
        
    except Exception as e:
        print(f"Error processing results: {e}")
        return 1

if __name__ == "__main__":
    exit(main())