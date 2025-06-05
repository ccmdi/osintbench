import re
import os
import argparse
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Answer:
    """Base class for all answer types"""
    pass

@dataclass
class LocationAnswer(Answer):
    lat: float
    lng: float

@dataclass
class JudgeAnswer(Answer):
    response: str

class TaskParser(ABC):
    @abstractmethod
    def parse(self, response: str) -> Answer:
        pass

class LocationParser(TaskParser):
    def parse(self, response: str) -> LocationAnswer:
        lat_match = re.search(r'lat:\s*([-+]?\d+\.?\d*)', response, re.IGNORECASE)
        lng_match = re.search(r'lng:\s*([-+]?\d+\.?\d*)', response, re.IGNORECASE)
        
        if not lat_match or not lng_match:
            raise ValueError("Missing lat/lng in structured format")
            
        lat = float(lat_match.group(1))
        lng = float(lng_match.group(1))
        
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            raise ValueError(f"Invalid coordinates: lat={lat}, lng={lng}")
            
        return LocationAnswer(lat=lat, lng=lng)

class JudgeParser(TaskParser):
    def parse(self, response: str) -> JudgeAnswer:
        return JudgeAnswer(response=response)

def get_parser(task_type: str) -> TaskParser:
    parsers = {
        'location': LocationParser(),
    }
    
    if task_type not in parsers:
        return JudgeParser()
    
    return parsers[task_type]

def parse_response(response: str, task) -> Answer:
    """Parse structured response based on task type"""
    parser = get_parser(task.type)
    return parser.parse(response)

def evaluate_answer(parsed_answer: Answer, task, case_id, run_folder = None) -> dict:   
    if task.type in ['location']:
        if isinstance(parsed_answer, LocationAnswer):
            import haversine
            distance_km = haversine.haversine(
                (parsed_answer.lat, parsed_answer.lng),
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
            print("JUDGEMENT")
            from scripts.judge import Judge
            judge = Judge()
            return judge.evaluate(parsed_answer.response, task, case_id, run_folder)
        except Exception as e:
            print(f"Judge evaluation failed: {e}")
            return {'score': 0.0, 'correct': False}
    
    return {'score': 0.0, 'correct': False}

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