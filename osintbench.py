import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import datetime
import argparse
from dotenv import load_dotenv

from scripts.parser import parse_response, Answer, evaluate_answer

SYSTEM_PROMPT_BASE = """
You are participating in an OSINT challenge. You are given task(s) that you must provide answers to using the provided evidence and any tools you have available.
For instance, you have access to Google Search, which may be required to answer the question. Take your time to reason through the evidence.

"""

SYSTEM_PROMPT_PRESTRUCTURE = """
Your final answer MUST be in structured format:

"""

SYSTEM_PROMPT_POSTSTRUCTURE = """
You must provide a structured answer for each task, BUT you should only provide a structured format for the task types you are given. For instance, do not provide a temporal task answer if there is not a temporal task.
"""

LOCATION_TASK_FORMAT = """
FOR LOCATION TASKS:
lat: [latitude as decimal number]
lng: [longitude as decimal number]
"""

IDENTIFICATION_TASK_FORMAT = """
FOR IDENTIFICATION TASKS:
type: [entity type - person/organization/vehicle/etc]
name: [specific name if identifiable]
"""

TEMPORAL_TASK_FORMAT = """
FOR TEMPORAL TASKS:
date: [YYYY-MM-DD or descriptive period]
time: [HH:MM or time period if applicable]
"""

ANALYSIS_TASK_FORMAT = """
FOR ANALYSIS TASKS:
conclusion: [conclusion to a question - must be ONE answer, no hedging]
"""

from models import *
load_dotenv()

@dataclass
class Task:
    task_id: int
    type: str
    prompt: str
    answer: Any

    @classmethod
    def from_dict(cls, data: dict) -> 'Task':
        return cls(
            task_id=data['id'],
            type=data['type'],
            prompt=data['prompt'],
            answer=data['answer']
        )

@dataclass
class Case:
    case_id: int
    images: List[str]
    info: str
    tasks: List[Task]

    @classmethod
    def from_dict(cls, data: dict, dataset_path: str = None) -> 'Case':
        images = data['images']
        if dataset_path:
            images = [os.path.join(dataset_path, img_path) for img_path in images]
        
        return cls(
            case_id=data['id'],
            images=images,
            info=data['info'],
            tasks=[Task.from_dict(task) for task in data['tasks']]
        )
@dataclass
class BenchmarkResult:
    case_obj: Case
    task_id: int
    answer: Answer = None
    parsed_answer: Answer = None
    evaluation: float = None
    refused: bool = False
    error_message: Optional[str] = None

class OsintBenchmark:
    def __init__(self, 
                 dataset_path: str,
                 model: str = "Claude4SonnetThinking",
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
        """Loads list of cases from the dataset."""
        with open(os.path.join(self.dataset_path, "metadata.json"), "r") as f:
            data = json.load(f)
            
        cases = []
        for case in data['cases']:
            cases.append(Case.from_dict(case, self.dataset_path))
        return cases

    def prompt(self, case: Case) -> str:
        """Builds prompt for a case."""
        prompt = SYSTEM_PROMPT_BASE + SYSTEM_PROMPT_PRESTRUCTURE
        for task in case.tasks:
            match task.type:
                case "location":
                    prompt += LOCATION_TASK_FORMAT
                case "identification":
                    prompt += IDENTIFICATION_TASK_FORMAT
                case "temporal":
                    prompt += TEMPORAL_TASK_FORMAT
                case "analysis":
                    prompt += ANALYSIS_TASK_FORMAT
        
        prompt += SYSTEM_PROMPT_POSTSTRUCTURE
        return prompt

    def run(self, args) -> Dict:
        """Runs the benchmark. Returns `_compile_results` output."""
        cases_to_test = self.cases

        if args.sample_id is not None:
            cases_to_test = [case for case in self.cases if case.case_id == args.sample_id]
            if not cases_to_test:
                raise ValueError(f"Case ID '{args.sample_id}' not found in dataset")
        elif args.samples and args.samples < len(self.cases):
            import random
            cases_to_test = random.sample(self.cases, args.samples)
                
        self.results = []
        
        for case in cases_to_test:
            print(f"Testing case: {case.case_id}")
            self._evaluate_case(case)
            
            self.save_results(run_folder + "/results/")
        
        return self._compile_results()
    
    def _evaluate_case(self, case: Case) -> None: #TODO: do we want to return anything?
        for attempt in range(self.max_retries):
            try:
                response = self.model.query(self.prompt(case), case, run_folder)
                
                os.makedirs(f"{run_folder}/output/", exist_ok=True)
                with open(f"{run_folder}/output/{case.case_id}.txt", "w", encoding="utf-8") as f:
                    f.write(response)
                
                try:

                    for task in case.tasks:
                        answer = parse_response(response, task)
                        evaluation = evaluate_answer(answer, task, case.case_id)
                        result = BenchmarkResult(case_obj=case, task_id=task.task_id, answer=task.answer, parsed_answer=answer, evaluation=evaluation)

                        self.results.append(result)
                        
                        if result.refused:
                            print(f"REFUSED: {result.error_message}")
                        else:
                            print("BOOM")
                            #TODO: ACCURACY
                            pass
                except ValueError as parse_error:
                    print(f"  Format error (attempt {attempt+1}): {str(parse_error)}")
                    if "missing required fields" in str(parse_error) or "parse" in str(parse_error):
                        return BenchmarkResult(
                            case_obj=case, 
                            answer=task.answer,
                            parsed_answer=None,
                            evaluation=None,
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
                    case_obj=case, 
                    answer=None, 
                    parsed_answer=None,
                    evaluation=None,
                    refused=True,
                    error_message=error_msg
                )
    
    def _compile_results(self) -> Dict:
        total = len(self.results)
        refusals = sum(1 for r in self.results if r.refused)
        
        # Now results are already task-level, so total_tasks = total results
        total_tasks = total
        
        # Count correct and sum scores from non-refused results with evaluations
        correct_tasks = sum(1 for r in self.results 
                        if not r.refused and r.evaluation and r.evaluation.get('correct', False))
        
        total_score = sum(r.evaluation.get('score', 0) for r in self.results 
                        if not r.refused and r.evaluation)
        
        valid_tasks = sum(1 for r in self.results if not r.refused and r.evaluation)
        
        return {
            "model": self.model.name,
            "test": os.path.basename(self.dataset_path),
            "n": len(set(r.case_obj.case_id for r in self.results)),  # Number of unique cases
            "total_tasks": total_tasks,
            "refusal_rate": refusals / total if total > 0 else 0,
            "avg_accuracy": total_score / valid_tasks if valid_tasks > 0 else 0,
            "accuracy_rate": correct_tasks / valid_tasks if valid_tasks > 0 else 0,
            "detailed_results": self.results
        }
    
    def save_results(self, output_path: str):
        results_dict = self._compile_results()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(f"{output_path}summary.json", "w") as f:
            json.dump({k: v for k, v in results_dict.items() if k != "detailed_results"}, f, indent=2)
        
        records = []
        for r in self.results:
            task = next(t for t in r.case_obj.tasks if t.task_id == r.task_id)
            records.append({
                "case_id": r.case_obj.case_id,
                "task_id": r.task_id,
                "task_type": task.type,
                "prompt": task.prompt,
                "answer": r.answer,
                "refused": r.refused,
                "error_message": r.error_message,
                "score": r.evaluation.get('score') if r.evaluation else None,
                "correct": r.evaluation.get('correct') if r.evaluation else None
            })
        
        pd.DataFrame(records).to_csv(f"{output_path}detailed.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSINT Benchmark Tool")
    parser.add_argument("--dataset", "-d", type=str, default="basic", 
                        help="Dataset subfolder to use (default: 'basic')")
    parser.add_argument("--samples", "-n", type=int, default=None,
                        help="Number of samples to test (default: all)")
    parser.add_argument("--sample-id", "-i", type=int, default=None, help="Run a specific sample by ID")
    parser.add_argument("--model", "-m", type=str, default="Claude3_7Sonnet",
                        help="Model to use (default: 'Claude3_7Sonnet')")
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
    
    results = benchmark.run(args)
    
    benchmark.save_results(run_folder + "/results/")
    
    print(f"Total samples: {results['n']}")
    print(f"Total tasks: {results['total_tasks']}")
    print(f"Refusal rate: {results['refusal_rate']:.2%}")
    print(f"Average accuracy: {results['avg_accuracy']:.3f}")
    print(f"Accuracy rate: {results['accuracy_rate']:.2%}")