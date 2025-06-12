import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import datetime
import argparse
from dotenv import load_dotenv

from scripts.eval import evaluate_answer, get_parser
from models import *
from prompt import get_prompt
from util import setup_logging, get_logger
from context import set_case, set_dataset_path, set_benchmark

load_dotenv()

logger = get_logger(__name__)

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
        images = data.get('images', [])
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
    task_type: str
    answer: Any = None
    parsed_answer: Any = None
    evaluation: float = None
    refused: bool = False
    error_message: Optional[str] = None

class OsintBenchmark:
    def __init__(self, 
                 dataset_path: str,
                 model: str,
                 api_key: Optional[str] = None,
                 max_retries: int = 3):
        self.dataset_path = dataset_path
        self.cases = self._load_dataset()
        self.results = []
        self.max_retries = max_retries
        
        try:
            if not model:
                raise ValueError("No model provided.")
            
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
        with open(os.path.join(self.dataset_path, "metadata.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
            
        cases = []
        for case in data['cases']:
            cases.append(Case.from_dict(case, self.dataset_path))
        return cases

    def run(self, args, run_folder: str) -> Dict:
        """Runs the benchmark. Returns `_compile_results` output."""
        cases_to_test = self.cases

        if args.sample_id is not None:
            cases_to_test = [case for case in self.cases if case.case_id == args.sample_id]
            if not cases_to_test:
                raise ValueError(f"Case ID '{args.sample_id}' not found in dataset")
        elif getattr(args, 'from', None) is not None:
            from_arg = getattr(args, 'from')
            
            # Parse range or single ID
            if ':' in from_arg:
                try:
                    from_id, to_id = map(int, from_arg.split(':'))
                    if from_id > to_id:
                        raise ValueError(f"Invalid range: start ID ({from_id}) must be <= end ID ({to_id})")
                    cases_to_test = [case for case in self.cases if from_id <= case.case_id <= to_id]
                    if not cases_to_test:
                        raise ValueError(f"No cases found with ID in range {from_id}:{to_id}")
                    logger.info(f"Testing cases from ID {from_id} to {to_id} ({len(cases_to_test)} cases)")
                except ValueError as e:
                    if "invalid literal" in str(e):
                        raise ValueError(f"Invalid range format '{from_arg}'. Use format like '5:10'")
                    raise
            else:
                try:
                    from_id = int(from_arg)
                    cases_to_test = [case for case in self.cases if case.case_id >= from_id]
                    if not cases_to_test:
                        raise ValueError(f"No cases found with ID >= {from_id}")
                    logger.info(f"Testing cases from ID {from_id} to end ({len(cases_to_test)} cases)")
                except ValueError:
                    raise ValueError(f"Invalid ID format '{from_arg}'. Use a number or range like '5:10'")
        elif args.samples and args.samples < len(self.cases):
            import random
            cases_to_test = random.sample(self.cases, args.samples)
            logger.info(f"Testing random sample of {args.samples} cases out of {len(self.cases)}")
        else:
            logger.info(f"Testing all {len(self.cases)} cases")
                
        self.results = []
        
        for i, case in enumerate(cases_to_test, 1):
            logger.announcement(f"Testing case {i}/{len(cases_to_test)}")
            set_case(case)
            self._evaluate_case(case, run_folder)
            self.save_results(run_folder + "/results/")
        
        logger.info("All cases completed, compiling results")
        return self._compile_results()
    
    def _evaluate_case(self, case: Case, run_folder: str) -> None:
        """Evaluates a case."""
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Querying model for case {case.case_id} (attempt {attempt+1})")
                response = self.model.query(get_prompt(case), run_folder)
                
                os.makedirs(f"{run_folder}/output/", exist_ok=True)
                with open(f"{run_folder}/output/{case.case_id}.txt", "w", encoding="utf-8") as f:
                    f.write(response)
                logger.debug(f"Saved response for case {case.case_id} to output file")
                
                try:
                    for task in case.tasks:
                        logger.debug(f"Processing task {task.task_id} ({task.type}) for case {case.case_id}")
                        parser = get_parser(task.type)
                        answer = parser.parse(response, task, case.case_id, run_folder)
                        evaluation = evaluate_answer(answer, task, case.case_id, run_folder)
                        evaluation['parser'] = parser.__class__.__name__

                        result = BenchmarkResult(
                            case_obj=case, 
                            task_id=task.task_id, 
                            task_type=task.type, 
                            answer=task.answer, 
                            parsed_answer=answer, 
                            evaluation=evaluation
                        )

                        self.results.append(result)
                        
                        if result.refused:
                            logger.warning(f"Task {task.task_id} refused: {result.error_message}")
                        else:
                            logger.debug(f"Task {task.task_id} completed successfully")
                    
                    logger.info(f"Case {case.case_id} completed successfully")
                    return  # Success - all tasks processed
                    
                except ValueError as parse_error:
                    logger.warning(f"Parse error for case {case.case_id} (attempt {attempt+1}): {str(parse_error)}")
                    if "missing required fields" in str(parse_error) or "parse" in str(parse_error):
                        for task in case.tasks:
                            error_result = BenchmarkResult(
                                case_obj=case, 
                                task_id=task.task_id,
                                task_type=task.type,
                                answer=task.answer,
                                parsed_answer=None,
                                evaluation=None,
                                refused=True,
                                error_message=f"Format error: {str(parse_error)}"
                            )
                            self.results.append(error_result)
                        logger.error(f"Case {case.case_id} failed with parse error (no retry)")
                        return  # Don't retry for format errors
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"API/network error for case {case.case_id} (attempt {attempt+1}): {error_msg}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying case {case.case_id}...")
                    continue
                
                # Final attempt failed - add error result for each task
                logger.error(f"Case {case.case_id} failed after {self.max_retries} attempts")
                for task in case.tasks:
                    error_result = BenchmarkResult(
                        case_obj=case, 
                        task_id=task.task_id,
                        task_type=task.type,
                        answer=task.answer, 
                        parsed_answer=None,
                        evaluation=None,
                        refused=True,
                        error_message=error_msg
                    )
                    self.results.append(error_result)
                return
    
    def _compile_results(self) -> Dict:        
        total = len(self.results)
        refusals = sum(1 for r in self.results if r.refused)
        
        total_tasks = total
        
        correct_tasks = sum(1 for r in self.results 
                        if not r.refused and r.evaluation and r.evaluation.get('correct', False))
        
        total_score = sum(r.evaluation.get('score', 0) for r in self.results 
                        if not r.refused and r.evaluation)
        
        valid_tasks = sum(1 for r in self.results if not r.refused and r.evaluation)

        location_tasks = sum(1 for r in self.results if not r.refused and r.evaluation and r.task_type == 'location')
        identification_tasks = sum(1 for r in self.results if not r.refused and r.evaluation and r.task_type == 'identification')
        temporal_tasks = sum(1 for r in self.results if not r.refused and r.evaluation and r.task_type == 'temporal')
        analysis_tasks = sum(1 for r in self.results if not r.refused and r.evaluation and r.task_type == 'analysis')

        location_score = sum(r.evaluation.get('score', 0) for r in self.results if not r.refused and r.evaluation and r.task_type == 'location')
        identification_score = sum(r.evaluation.get('score', 0) for r in self.results if not r.refused and r.evaluation and r.task_type == 'identification')
        temporal_score = sum(r.evaluation.get('score', 0) for r in self.results if not r.refused and r.evaluation and r.task_type == 'temporal')
        analysis_score = sum(r.evaluation.get('score', 0) for r in self.results if not r.refused and r.evaluation and r.task_type == 'analysis')
        
        results_dict = {
            "model": self.model.name,
            "test": os.path.basename(self.dataset_path),
            "n": len(set(r.case_obj.case_id for r in self.results)),
            "total_tasks": total_tasks,
            "refusal_rate": refusals / total if total > 0 else 0,
            "location_accuracy": location_score / location_tasks if location_tasks > 0 else 0,
            "identification_accuracy": identification_score / identification_tasks if identification_tasks > 0 else 0,
            "temporal_accuracy": temporal_score / temporal_tasks if temporal_tasks > 0 else 0,
            "analysis_accuracy": analysis_score / analysis_tasks if analysis_tasks > 0 else 0,
            "overall_accuracy": total_score / valid_tasks if valid_tasks > 0 else 0,
            "task_accuracy": correct_tasks / valid_tasks if valid_tasks > 0 else 0,
            "tools": self.model.get_tools(),
            "provider": self.model.provider,
            "detailed_results": self.results
        }
        
        return results_dict
    
    def save_results(self, output_path: str):
        logger.debug(f"Saving results to: {output_path}")
        
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
                "refused": r.refused,
                "error_message": r.error_message,
                "parser": r.evaluation.get('parser') if r.evaluation else None,
                "score": r.evaluation.get('score') if r.evaluation else None,
                "correct": r.evaluation.get('correct') if r.evaluation else None,
            })
        
        pd.DataFrame(records).to_csv(f"{output_path}detailed.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSINT Benchmark Tool")
    parser.add_argument("--dataset", "-d", type=str, default="basic", 
                        help="Dataset subfolder to use (default: 'basic')")
    parser.add_argument("--samples", "-n", type=int, default=None,
                        help="Number of samples to test (default: all)")
    parser.add_argument("--sample-id", "-i", type=int, default=None, help="Run a specific sample by ID")
    parser.add_argument("--from", "-f", type=str, default=None, help="Start from a specific sample ID and run to the end, or specify a range like '5:10'")
    parser.add_argument("--model", "-m", type=str, help="Model to use")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Maximum number of retries for API/network errors (default: 3)")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")
    args = parser.parse_args()
    
    dataset_path = f"dataset/{args.dataset}"

    benchmark = OsintBenchmark(
        dataset_path=dataset_path,
        model=args.model,
        max_retries=args.max_retries
    )
    set_benchmark(benchmark)
    set_dataset_path(benchmark.dataset_path)

    runtime = datetime.datetime.now().strftime('%Y-%m-%dT%H_%M_%S')
    run_folder = f"responses/{benchmark.model.name}_{args.dataset}_{runtime}"
    
    log_file = setup_logging(run_folder, args.log_level)
    
    logger.announcement(f"Starting OSINT Benchmark - Model: {benchmark.model.name}, Dataset: {args.dataset}, Tools: {len(benchmark.model.get_tools())}")
    logger.info(f"Run folder: {run_folder}")
    logger.info(f"Log file: {log_file}")
    
    results = benchmark.run(args, run_folder)
    
    benchmark.save_results(run_folder + "/results/")
    
    print(f"Total samples: {results['n']}")
    print(f"Total tasks: {results['total_tasks']}")
    print(f"Refusal rate: {results['refusal_rate']:.2%}")
    print(f"Location accuracy: {results['location_accuracy']:.3f}")
    print(f"Identification accuracy: {results['identification_accuracy']:.3f}")
    print(f"Temporal accuracy: {results['temporal_accuracy']:.3f}")
    print(f"Analysis accuracy: {results['analysis_accuracy']:.3f}")
    print(f"Overall accuracy: {results['overall_accuracy']:.3f}")
    print(f"Task accuracy: {results['task_accuracy']:.3f}")
    
    logger.info("OSINT Benchmark completed successfully")