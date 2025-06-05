#TODO: EVERYTHING

# fix.py
# Update the CSV and summary to be correct from edited results

import pandas as pd
import os
import json
import argparse
from typing import Dict
import haversine

def compile_summary(df: pd.DataFrame) -> Dict:
    """Compile benchmark results into a summary dictionary"""
    total = len(df)
    refusals = sum(1 for _, r in df.iterrows() if r.get('refused', False))
    
    valid_df = df[~df['refused'].fillna(False)]
    avg_distance = valid_df['distance_km'].mean() if not valid_df.empty else None

    median_distance = valid_df['distance_km'].median() if not valid_df.empty and 'distance_km' in valid_df.columns else None
    
    return {
        "n": total,
        "refusal_rate": refusals / total if total > 0 else 0,
        "average_distance_km": avg_distance,
        "median_distance_km": median_distance,
    }

def fill_missing_data(csv_path, metadata_path, output_dir=None, summary_only=False):
    """Fill in missing distance_km t fields and generate summary"""
    if output_dir is None:
        output_dir = os.path.dirname(csv_path) or '.'
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_csv = os.path.join(output_dir, 'detailed.csv')
    output_json = os.path.join(output_dir, 'summary.json')
    
    df = pd.read_csv(csv_path)
    
    # For each row, calculate missing values if possible
    for index, row in df.iterrows():
        if row.get('refused', False) == True:
            continue
        
        # Make sure we have the required coordinates
        if pd.isna(row.get('lat_true')) or pd.isna(row.get('lng_true')) or \
           pd.isna(row.get('lat_guess')) or pd.isna(row.get('lng_guess')):
            continue
        
        true_coords = (row['lat_true'], row['lng_true'])
        guess_coords = (row['lat_guess'], row['lng_guess'])
        
        distance_km = haversine.haversine(true_coords, guess_coords)
        df.at[index, 'distance_km'] = distance_km

    summary = compile_summary(df)

    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {output_json}")
    
    # Only save the CSV if not in summary_only mode
    if not summary_only:
        df.to_csv(output_csv, index=False)
        print(f"Updated data saved to {output_csv}")
    
    return df, summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fill missing geolocation data and generate summary")
    parser.add_argument("csv_path", type=str, help="Path to CSV file")
    parser.add_argument("--metadata", "-m", type=str, required=True,
                        help="Path to metadata.json file")
    parser.add_argument("--output-dir", "-o", type=str, default=None, 
                        help="Output directory for results (default: same as input)")
    parser.add_argument("--summary-only", "-s", action="store_true",
                        help="Only generate the summary.json file without the detailed CSV")
    
    args = parser.parse_args()
    
    df, summary = fill_missing_data(args.csv_path, args.metadata, args.output_dir, args.summary_only)
    
    # Print summary
    print("\nBenchmark Results:")
    print(f"Total samples: {summary['n']}")
    print(f"Average distance: {summary['average_distance_km']:.1f} km")
    print(f"Refusal rate: {summary['refusal_rate']:.2%}")