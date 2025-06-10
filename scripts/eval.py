import re
import os
import argparse
from typing import Dict, Any

import haversine

from models import Gemini2Flash
from prompt import system_prompt
from util import get_logger


logger = get_logger(__name__)

class Judge:
    def __init__(self):
        """Initialize the judge with a small, fast model"""
        self.model_class = Gemini2Flash
        self.api_key = os.getenv(self.model_class.api_key_name)
        self.model = self.model_class(self.api_key)
    
    def parse_location_response(self, response: str, task, case_id, run_folder = None) -> Dict[str, Any]:
        try:
            judge_parse_prompt = f"""Parse the following response into a JSON object containing only the fields 'lat' and 'lng' (latitude and longitude).
            The latitude and longitude should be in decimal degrees format (e.g. 37.774929, -122.419418).
            If either the latitude or longitude are missing, return None. If the response does not contain coordinates at all, return None.

            The format should look exactly like this:
            {{
                "lat": <latitude>,
                "lng": <longitude>
            }}

            A None response should look like this:
            {{
                "lat": null,
                "lng": null
            }}

        Response to parse: {response}
        """
            judge_response = self.model.query(system_prompt(judge_parse_prompt))

            if run_folder:
                os.makedirs(f"{run_folder}/judge", exist_ok=True)
                with open(f"{run_folder}/judge/{case_id}_parse_{task.task_id}.txt", "w") as f:
                    f.write(judge_response)

            return judge_response
        except Exception as e:
            pass
    
    def evaluate(self, response: str, task, case_id, run_folder = None) -> Dict[str, Any]:
        """
        Evaluate a response against the task and ground truth using a language model
        
        Args:
            response: The model's response to evaluate
            task: The original task
            ground_truth: Dictionary containing the correct answer
            
        Returns:
            Dictionary with 'correct' (bool) and 'reasoning' (str)
        """
                
        judge_eval_prompt = f"""You are an expert evaluator. Your job is to determine if a response correctly answers the given task.

        Task Type: {task.type}
        Task: {task.prompt}

        Correct Answer: {task.answer}

        Response to Evaluate: {response}

        Instructions:
        1. Carefully compare the response against the correct answer
        2. For location tasks: Check if coordinates are very close (within ~1km is acceptable)
        3. For identification tasks: Check if the type and name match (partial matches acceptable)
        4. For temporal tasks: Check if dates/times match reasonably
        5. For analysis tasks: Check if the conclusion is reasonable and matches

        If the response contains hedging (i.e. the model cannot decide between multiple answers), you should return NO.

        Respond with EXACTLY this format:
        CORRECT: YES/NO
        REASONING: Brief explanation of why it's correct or incorrect"""

        try:
            judge_response = self.model.query(system_prompt(judge_eval_prompt))
            
            if run_folder:
                os.makedirs(f"{run_folder}/judge", exist_ok=True)
                with open(f"{run_folder}/judge/{case_id}_{task.task_id}.txt", "w", encoding="utf-8") as f:
                    f.write(judge_response)

            return self._parse_judge_response(judge_response)
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            return {"correct": False, "reasoning": f"Evaluation error: {str(e)}"}
    
    def _parse_judge_response(self, response: str) -> Dict[str, Any]:
        """Parse the judge's response"""
        lines = response.strip().split('\n')
        correct = False
        reasoning = "Could not parse judge response"
        
        for line in lines:
            line = line.strip()
            if line.startswith('CORRECT:'):
                correct_text = line.replace('CORRECT:', '').strip().upper()
                correct = correct_text == 'YES'
            elif line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()
        
        logger.evaluation("CORRECT?: " + str(correct) + " - REASONING: " + str(reasoning))
        
        return {"correct": correct, "score": 1 if correct else 0, "reasoning": reasoning}


class JudgeParser:
    def parse(self, response: str, task, case_id, run_folder = None) -> dict:
        return response

class LocationParser(JudgeParser):
    def parse(self, response: str, task, case_id, run_folder = None) -> dict:
        lat_match = re.search(r'lat:\s*([-+]?\d+\.?\d*)', response, re.IGNORECASE)
        lng_match = re.search(r'lng:\s*([-+]?\d+\.?\d*)', response, re.IGNORECASE)
        
        # Regex parse
        if lat_match and lng_match:
            logger.evaluation(self.__class__.__name__ + " - Parsing coordinates.")
            lat = float(lat_match.group(1))
            lng = float(lng_match.group(1))
            
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                raise ValueError(f"Invalid coordinates: lat={lat}, lng={lng}")
        # Judge parse
        else:
            logger.evaluation(self.__class__.__name__ + " - Defaulting to judge parsing.")
            try:
                judge = Judge()
                judge_response = judge.parse_location_response(response, task, case_id, run_folder)
            except Exception as e:
                logger.error(f"Judge parsing failed: {e}")
                return {'lat': None, 'lng': None}

            if judge_response:
                import json
                try:
                    judge_response = judge_response.split("```json")[1].split("```")[0]
                    parsed_json = json.loads(judge_response)

                    lat = parsed_json['lat']
                    lng = parsed_json['lng']
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON response from judge")
            else:
                return {'lat': None, 'lng': None}
        
        return {
            'lat': lat,
            'lng': lng
        }

def get_parser(task_type: str) -> Any:
    if task_type == 'location':
        return LocationParser()
    else:
        return JudgeParser()

def evaluate_answer(parsed_answer, task, case_id, run_folder = None) -> dict:   
    if task.type in ['location']:
        if isinstance(parsed_answer, dict) and parsed_answer is not None:
            if parsed_answer.get('lat') is None or parsed_answer.get('lng') is None:
                return {'score': 0.0, 'correct': False, 'refused': True}
            
            distance_km = haversine.haversine(
                (parsed_answer['lat'], parsed_answer['lng']),
                (task.answer['lat'], task.answer['lng'])
            )
            
            if distance_km <= 0.1:
                score = 1.0
            elif distance_km <= 0.5:
                score = 0.8
            elif distance_km <= 1:
                score = 0.5
            elif distance_km <= 10:
                score = 0.2
            else:
                score = 0.0
            
            logger.evaluation(f"Score: {score}, Correct: {score >= 0.8}, Distance: {round(distance_km, 2)} km")
            evaluation = {
                'score': score,
                'correct': score >= 0.8,
                'distance_km': distance_km
            }
    else:
        try:
            judge = Judge()
            evaluation = judge.evaluate(parsed_answer, task, case_id, run_folder)
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            evaluation = {'score': 0.0, 'correct': False}
    
    return evaluation or {'score': 0.0, 'correct': False, 'refused': True}

if __name__ == "__main__":
    import json
    from dotenv import load_dotenv
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Re-evaluate existing model responses using current parsers and judges.")
    parser.add_argument("response_folder", type=str, help="Path to the response folder (e.g., 'responses/Claude_4_Sonnet_basic_2024-01-15T10_30_45')")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset path to get ground truth (e.g., 'dataset/basic')")
    args = parser.parse_args()
    
    # Load dataset metadata for ground truth
    metadata_path = os.path.join(args.dataset, "metadata.json")
    if not os.path.exists(metadata_path):
        logger.error(f"Error: Dataset metadata '{metadata_path}' not found.")
        exit(1)
    
    with open(metadata_path, "r") as f:
        dataset = json.load(f)
    
    # Create lookup for tasks by case_id and task_id
    task_lookup = {}
    for case_data in dataset['cases']:
        case_id = case_data['id']
        for task_data in case_data['tasks']:
            task_id = task_data['id']
            task_lookup[(case_id, task_id)] = {
                'type': task_data['type'],
                'prompt': task_data['prompt'],
                'answer': task_data['answer']
            }
    
    output_dir = os.path.join(args.response_folder, "output")
    if not os.path.isdir(output_dir):
        logger.error(f"Error: Output directory '{output_dir}' not found.")
        exit(1)

    # Load existing detailed results to get task information
    results_path = os.path.join(args.response_folder, "results", "detailed.csv")
    if not os.path.exists(results_path):
        logger.error(f"Error: Results file '{results_path}' not found.")
        exit(1)
    
    import pandas as pd
    existing_results = pd.read_csv(results_path)
    
    valid_responses = 0
    total_files = 0
    re_evaluation_results = []

    for filename in os.listdir(output_dir):
        if filename.endswith(".txt"):
            total_files += 1
            case_id = int(os.path.splitext(filename)[0])  # Extract case_id from filename
            file_path = os.path.join(output_dir, filename)
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    response_content = f.read()
                
                # Get all tasks for this case from existing results
                case_tasks = existing_results[existing_results['case_id'] == case_id]
                
                for _, row in case_tasks.iterrows():
                    task_id = row['task_id']
                    task_type = row['task_type']
                    
                    # Get ground truth from task_lookup
                    if (case_id, task_id) not in task_lookup:
                        logger.warning(f"Warning: Task {task_id} for case {case_id} not found in dataset")
                        continue
                    
                    ground_truth = task_lookup[(case_id, task_id)]
                    
                    try:
                        # Create simple task object with dot notation access
                        from types import SimpleNamespace
                        task = SimpleNamespace(
                            task_id=task_id,
                            type=task_type,
                            prompt=ground_truth['prompt'],
                            answer=ground_truth['answer']
                        )
                        
                        # Parse the response
                        parser = get_parser(task_type)
                        parsed_answer = parser.parse(response_content, task, case_id, args.response_folder)
                        
                        # Evaluate the parsed answer
                        evaluation = evaluate_answer(parsed_answer, task, case_id, args.response_folder)
                        
                        re_evaluation_results.append({
                            "case_id": case_id,
                            "task_id": task_id,
                            "task_type": task_type,
                            "prompt": task.prompt,
                            "refused": evaluation.get('refused', False),
                            "error_message": evaluation.get('error_message', None),
                            "parser": parser.__class__.__name__,
                            "score": evaluation.get('score', 0),
                            "correct": evaluation.get('correct', False)
                        })
                        
                        logger.info(f"Case {case_id}, Task {task_id} ({task_type}): Score {evaluation.get('score', 0):.2f}, Correct: {evaluation.get('correct', False)}")
                        
                        if evaluation.get('correct', False):
                            valid_responses += 1
                            
                    except Exception as parse_error:
                        logger.error(f"Error parsing/evaluating case {case_id}, task {task_id}: {parse_error}")
                        re_evaluation_results.append({
                            "case_id": case_id,
                            "task_id": task_id,
                            "task_type": task_type,
                            "prompt": task.prompt,
                            "refused": True,
                            "error_message": str(parse_error),
                            "parser": parser.__class__.__name__,
                            "score": 0,
                            "correct": False,
                        })
                        
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")

    # Save re-evaluation results
    re_eval_df = pd.DataFrame(re_evaluation_results)
    re_eval_path = os.path.join(args.response_folder, "results", "re_evaluation.csv")
    re_eval_df.to_csv(re_eval_path, index=False)
    
    print(f"\nRe-evaluation complete!")
    print(f"Processed {total_files} response files")
    print(f"Total task evaluations: {len(re_evaluation_results)}")
    print(f"Correct answers: {valid_responses}")
    print(f"Overall accuracy: {valid_responses/len(re_evaluation_results)*100:.1f}%" if re_evaluation_results else "0%")
    
    # Show comparison with original results
    if len(re_evaluation_results) > 0:
        original_correct = sum(1 for _, row in existing_results.iterrows() if row.get('correct'))
        new_correct = sum(1 for r in re_evaluation_results if r['correct'])
        print(f"\nComparison:")
        print(f"Original correct: {original_correct}")
        print(f"New correct: {new_correct}")
        print(f"Difference: {new_correct - original_correct:+d}")