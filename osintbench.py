import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import datetime
import argparse
from dotenv import load_dotenv

from scripts.parser import parse_response, evaluate_answer

SYSTEM_PROMPT = """
<system>
You are participating in an OSINT (Open Source Intelligence) challenge.

Take your time to reason through the evidence. Your final answer MUST be in structured format:

FOR LOCATION TASKS:
lat: [latitude as decimal number]
lng: [longitude as decimal number]

FOR IDENTIFICATION TASKS:
type: [entity type - person/organization/vehicle/etc]
name: [specific name if identifiable]

FOR TEMPORAL TASKS:
date: [YYYY-MM-DD or descriptive period]
time: [HH:MM or time period if applicable]

FOR ANALYSIS TASKS:
conclusion: [main conclusion]
evidence: [key evidence points]

You can provide reasoning above the structured answer, but the structured format MUST be included.

You have access to Google Search, which you should use to improve your answer.
</system>
"""

from models import *

load_dotenv()

@dataclass
class Task:
    type: str
    prompt: str
    answer: Any

@dataclass
class Case:
    case_id: int
    images: List[str]
    info: str
    tasks: List[Task]
    
    @property
    def coordinates(self) -> Tuple[float, float]:
        return (self.lat, self.lng)
    
@dataclass
class BenchmarkResult:
    case_obj: Case
    answers: Dict[Task, Any]
    accuracy: float = 0.0
    refused: bool = False
    error_message: Optional[str] = None
    
    def calculate_metrics(self):
        if self.refused or self.answers is None:
            return
        
        for task in self.case_obj.tasks:
            answer = self.answers[task]
            ground_truth = task.answer
            self.accuracy += evaluate_answer(answer, ground_truth, task.type)

class OsintBenchmark:
    def __init__(self, 
                 dataset_path: str,
                 model: str = "ClaudeHaiku",
                 api_key: Optional[str] = None,
                 max_retries: int = 3):
        self.dataset_path = dataset_path
        self.cases = self._load_dataset()
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
        
    def _load_dataset(self) -> List[Case]:
        with open(os.path.join(self.dataset_path, "metadata.json"), "r") as f:
            data = json.load(f)
            
        cases = []
        for item in data['tasks']:
            cases.append(Case(
                case_id=item["id"],
                images=item["images"],
                info=item["info"],
                tasks=item["tasks"]
            ))
        return cases
    
    def run_benchmark(self, args) -> Dict:
        cases_to_test = self.cases

        if args.sample_id is not None:
            cases_to_test = [case for case in self.cases if case.case_id == str(args.sample_id)]
            if not cases_to_test:
                raise ValueError(f"Case ID '{args.sample_id}' not found in dataset")
        elif args.samples and args.samples < len(self.cases):
            import random
            cases_to_test = random.sample(self.cases, args.samples)
                
        self.results = []
        
        for case in cases_to_test:
            print(f"Testing case: {case.case_id}")
            result = self._evaluate_case(case)
            self.results.append(result)
            
            if result.refused:
                print(f"REFUSED: {result.error_message}")
            else:
                #TODO: ACCURACY
                pass
            
            self.save_results(run_folder + "/results/")
        
        return self._compile_results()
    
    def _evaluate_case(self, case: Case) -> BenchmarkResult:
        for attempt in range(self.max_retries):
            try:
                response = self.model.query(SYSTEM_PROMPT, case, run_folder)
                
                os.makedirs(f"{run_folder}/output/", exist_ok=True)
                
                with open(f"{run_folder}/output/{case.case_id}.txt", "w", encoding="utf-8") as f:
                    f.write(response)
                
                try:
                    answers = []
                    for task in case.tasks:
                        answers.append(parse_response(response, task.type))
                    
                    result = BenchmarkResult(case=case, answers=answers)
                    result.calculate_metrics()
                    return result
                except ValueError as parse_error:
                    print(f"  Format error (attempt {attempt+1}): {str(parse_error)}")
                    if "missing required fields" in str(parse_error) or "parse" in str(parse_error):
                        return BenchmarkResult(
                            case=case, 
                            answers=None, 
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
                    case=case, 
                    answers=None, 
                    refused=True,
                    error_message=error_msg
                )
        
        return BenchmarkResult(
            case=case, 
            answers=None,
            refused=True,
            error_message="Max retries exceeded"
        )
    
    def _compile_results(self) -> Dict:
        total = len(self.results)
        refusals = sum(1 for r in self.results if r.refused)
        
        return {
            "model": self.model.name,
            "test": os.path.basename(self.dataset_path),
            "n": total,
            "refusal_rate": refusals / total if total > 0 else 0,
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
                "case_id": r.case.case_id,
                "accuracy": r.accuracy,
                "refused": r.refused,
                "error_message": r.error_message
            }
                
            records.append(record)
            
        pd.DataFrame(records).to_csv(f"{output_path}detailed.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSINT Benchmark Tool")
    parser.add_argument("--dataset", "-d", type=str, default="basic", 
                        help="Dataset subfolder to use (default: 'basic')")
    parser.add_argument("--samples", "-n", type=int, default=None,
                        help="Number of samples to test (default: all)")
    parser.add_argument("--sample-id", "-i", type=int, default=None, help="Run a specific sample by ID")
    parser.add_argument("--model", "-m", type=str, default="Gemini2_5Pro",
                        help="Model to use (default: 'Gemini2_5Pro')")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Maximum number of retries for API/network errors (default: 3)")
    args = parser.parse_args()
    
    dataset_path = f"dataset/{args.dataset}"

    benchmark = OsintBenchmark(
        dataset_path=dataset_path,
        model=args.model,
        max_retries=args.max_retries
    )

    runtime = datetime.datetime.now().strftime('%Y-%m-%dT%H_%M_%S')
    run_folder = f"responses/{benchmark.model.name}_{args.dataset}_{runtime}"
    
    results = benchmark.run_benchmark(args)
    
    benchmark.save_results(run_folder + "/results/")
    
    print(f"Total samples: {results['n']}")
    print(f"Refusal rate: {results['refusal_rate']:.2%}")