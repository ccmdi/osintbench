import re
import os
import argparse
from abc import ABC, abstractmethod
from typing import Dict, Any

import haversine

from models import Gemini2Flash
from prompt import system_prompt

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
                with open(f"{run_folder}/judge/{case_id}_{task.task_id}.txt", "w") as f:
                    f.write(judge_response)

            return self._parse_judge_response(judge_response)
        except Exception as e:
            print(f"Error during model evaluation: {e}")
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
        
        print(correct, reasoning)
        
        return {"correct": correct, "score": 1 if correct else 0, "reasoning": reasoning}



class LocationParser:
    def parse(self, response: str, task, case_id, run_folder = None) -> dict:
        lat_match = re.search(r'lat:\s*([-+]?\d+\.?\d*)', response, re.IGNORECASE)
        lng_match = re.search(r'lng:\s*([-+]?\d+\.?\d*)', response, re.IGNORECASE)
        
        # Regex parse
        if lat_match and lng_match:            
            lat = float(lat_match.group(1))
            lng = float(lng_match.group(1))
            
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                raise ValueError(f"Invalid coordinates: lat={lat}, lng={lng}")
        # Judge parse
        else:
            print(self.__class__.__name__ + ": Defaulting to judge parsing.")
            try:
                judge = Judge()
                judge_response = judge.parse_location_response(response, task, case_id, run_folder)
            except Exception as e:
                print(f"Judge parsing failed: {e}")
                return None

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
                return None
        
        return {
            'lat': lat,
            'lng': lng
        }

def parse_response(response: str, task, case_id, run_folder = None) -> Any:
    """Parse structured response based on task type"""
    if task.type == 'location':
        parser = LocationParser()
        return parser.parse(response, task, case_id, run_folder)
    else:
        return response

def evaluate_answer(parsed_answer, task, case_id, run_folder = None) -> dict:   
    if task.type in ['location']:
        if isinstance(parsed_answer, dict):
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
                
            return {
                'score': score,
                'correct': score >= 0.8,
                'distance_km': distance_km
            }
    else:
        try:
            judge = Judge()
            return judge.evaluate(parsed_answer, task, case_id, run_folder)
        except Exception as e:
            print(f"Judge evaluation failed: {e}")
            return {'score': 0.0, 'correct': False}
    
    return {'score': 0.0, 'correct': False, 'refused': True}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse model responses and count valid ones.")
    parser.add_argument("response_folder", type=str, help="Path to the response folder containing an 'output' subdirectory with .txt files.")
    args = parser.parse_args()

    output_dir = os.path.join(args.response_folder, "output")
    if not os.path.isdir(output_dir):
        print(f"Error: Output directory '{output_dir}' not found.")
        exit(1)

    valid_responses = 0
    total_files = 0

    for filename in os.listdir(output_dir):
        if filename.endswith(".txt"):
            total_files += 1
            file_path = os.path.join(output_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                g = parse_response(content)
                print(os.path.splitext(os.path.basename(filename))[0], g.lat, g.lng)
                valid_responses += 1
            except ValueError as e:
                print(f"Invalid response in {filename}: {e}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Found {valid_responses} valid responses out of {total_files} files in {output_dir}") 