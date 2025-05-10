import os
import json
import math
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import datetime
import argparse
import haversine
from dotenv import load_dotenv

from scripts.parser import parse_response, Guess

SYSTEM_PROMPT = """
You are participating in a geolocation challenge.

Take your time to reason through the evidence. Your final answer MUST include these three lines somewhere in your response:

lat: [latitude as a decimal number]
lng: [longitude as a decimal number]

You can provide additional reasoning or explanation, but these three specific lines MUST be included.

 You have access to Google Search, which you should use to improve your answer.
"""

from models import *

load_dotenv()

@dataclass
class Location:
    location_id: int
    images: List[str]
    lat: float
    lng: float
    
    @property
    def coordinates(self) -> Tuple[float, float]:
        return (self.lat, self.lng)
    
@dataclass
class BenchmarkResult:
    location: Location
    guess: Optional[Guess]
    distance_km: Optional[float] = None
    refused: bool = False
    error_message: Optional[str] = None
    
    def calculate_metrics(self):
        if self.refused or self.guess is None:
            self.distance_km = None
            return
            
        self.distance_km = haversine.haversine(
            self.location.coordinates, 
            self.guess.coordinates
        )

class GeoGuessrBenchmark:
    def __init__(self, 
                 dataset_path: str,
                 model: str = "ClaudeHaiku",
                 api_key: Optional[str] = None,
                 max_retries: int = 3):
        self.dataset_path = dataset_path
        self.locations = self._load_dataset()
        self.results = []
        self.max_retries = max_retries
        
        try:
            model_class = globals()[model]
            
            if not issubclass(model_class, BaseMultimodalModel):
                raise ValueError(f"{model} is not a valid BaseMultimodalModel class")
                
            if not api_key and model_class.api_key_name:
                api_key = os.getenv(model_class.api_key_name)
                
            if not api_key:
                raise ValueError(f"API key {model_class.api_key_name} not found for {model}")
                
            self.model = model_class(api_key)
            
        except KeyError:
            raise ValueError(f"Unknown model provider: {model}. Make sure the class is defined.")
        
    def _load_dataset(self) -> List[Location]:
        with open(os.path.join(self.dataset_path, "metadata.json"), "r") as f:
            data = json.load(f)
            
        locations = []
        for item in data['images']:
            locations.append(Location(
                location_id=item["id"],
                images=item["images"],
                lat=item["lat"],
                lng=item["lng"]
            ))
        return locations
    
    def run_benchmark(self, args) -> Dict:
        locations_to_test = self.locations

        if args.sample_id is not None:
            locations_to_test = [loc for loc in self.locations if loc.location_id == str(args.sample_id)]
            if not locations_to_test:
                raise ValueError(f"Location ID '{args.sample_id}' not found in dataset")
        elif args.samples and args.samples < len(self.locations):
            import random
            locations_to_test = random.sample(self.locations, args.samples)
                
        self.results = []
        
        for location in locations_to_test:
            print(f"Testing location: {location.location_id}")
            result = self._evaluate_location(location)
            self.results.append(result)
            
            if result.refused:
                print(f"REFUSED: {result.error_message}")
            else:
                distance = f"{result.distance_km:.1f}km" if result.distance_km is not None else "N/A"
                print(f"Distance: {distance}")
            
            self._save_incremental_results(run_folder + "/results/")
        
        return self._compile_results()
    
    def _evaluate_location(self, location: Location) -> BenchmarkResult:
        for attempt in range(self.max_retries):
            try:
                response = self.model.query(location.images, SYSTEM_PROMPT, run_folder, location.location_id)
                
                os.makedirs(f"{run_folder}/output/", exist_ok=True)
                
                with open(f"{run_folder}/output/{location.location_id}.txt", "w", encoding="utf-8") as f:
                    f.write(response)
                
                try:
                    guess = parse_response(response)
                    result = BenchmarkResult(location=location, guess=guess)
                    result.calculate_metrics()
                    return result
                except ValueError as parse_error:
                    print(f"  Format error (attempt {attempt+1}): {str(parse_error)}")
                    if "missing required fields" in str(parse_error) or "parse" in str(parse_error):
                        return BenchmarkResult(
                            location=location, 
                            guess=None, 
                            refused=True,
                            error_message=f"Format error: {str(parse_error)}"
                        )
                
            except Exception as e:
                error_msg = str(e)
                print(f"  API/network error (attempt {attempt+1}): {error_msg}")
                if attempt < self.max_retries - 1:
                    print(f"  Retrying...")
                    continue
                
                return BenchmarkResult(
                    location=location, 
                    guess=None, 
                    refused=True,
                    error_message=error_msg
                )
        
        return BenchmarkResult(
            location=location, 
            guess=None, 
            refused=True,
            error_message="Max retries exceeded"
        )
    
    def _compile_results(self) -> Dict:
        total = len(self.results)
        refusals = sum(1 for r in self.results if r.refused)
        
        valid_results = [r for r in self.results if not r.refused]
        avg_distance = sum(r.distance_km for r in valid_results) / len(valid_results) if valid_results else None
        median_distance = sorted(r.distance_km for r in valid_results)[len(valid_results) // 2] if valid_results else None
        
        return {
            "model": self.model.name,
            "test": os.path.basename(self.dataset_path),
            "n": total,
            "refusal_rate": refusals / total if total > 0 else 0,
            "average_distance_km": avg_distance,
            "median_distance_km": median_distance,
            "detailed_results": self.results
        }
    
    def save_results(self, output_path: str):
        results_dict = self._compile_results()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(f"{output_path}summary.json", "w") as f:
            json.dump({k: v for k, v in results_dict.items() if k != "detailed_results"}, f, indent=2)
        
        records = []
        for r in self.results:
            record = {
                "location_id": r.location.location_id,
                "lat_true": r.location.lat,
                "lng_true": r.location.lng,
                "refused": r.refused,
                "error_message": r.error_message
            }
            
            if not r.refused and r.guess:
                record.update({
                    "lat_guess": r.guess.lat,
                    "lng_guess": r.guess.lng,
                    "distance_km": r.distance_km
                })
                
            records.append(record)
            
        pd.DataFrame(records).to_csv(f"{output_path}detailed.csv", index=False)
    
    def _save_incremental_results(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        records = []
        for r in self.results:
            record = {
                "location_id": r.location.location_id,
                "lat_true": r.location.lat,
                "lng_true": r.location.lng,
                "refused": r.refused,
                "error_message": r.error_message
            }
            
            if not r.refused and r.guess:
                record.update({
                    "lat_guess": r.guess.lat,
                    "lng_guess": r.guess.lng,
                    "distance_km": r.distance_km
                })
                
            records.append(record)
            
        pd.DataFrame(records).to_csv(f"{output_path}detailed.csv", index=False)
        
        results_dict = self._compile_results()
        with open(f"{output_path}summary.json", "w") as f:
            json.dump({k: v for k, v in results_dict.items() if k != "detailed_results"}, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoGuessr Benchmark Tool")
    parser.add_argument("--dataset", "-d", type=str, default="acw", 
                        help="Dataset subfolder to use (default: 'acw')")
    parser.add_argument("--samples", "-n", type=int, default=None,
                        help="Number of samples to test (default: all)")
    parser.add_argument("--sample-id", "-i", type=int, default=None, help="Run a specific sample by ID")
    parser.add_argument("--model", "-m", type=str, default="claude",
                        help="Model provider to use (default: 'claude')")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Maximum number of retries for API/network errors (default: 3)")
    args = parser.parse_args()
    
    dataset_path = f"dataset/{args.dataset}"

    benchmark = GeoGuessrBenchmark(
        dataset_path=dataset_path,
        model=args.model,
        max_retries=args.max_retries
    )

    runtime = datetime.datetime.now().strftime('%Y-%m-%dT%H_%M_%S')
    run_folder = f"responses/{benchmark.model.name}_{args.dataset}_{runtime}"
    
    results = benchmark.run_benchmark(args)
    
    benchmark.save_results(run_folder + "/results/")
    
    print(f"Total samples: {results['n']}")
    print(f"Average distance: {results['average_distance_km']:.1f} km")
    print(f"Refusal rate: {results['refusal_rate']:.2%}")