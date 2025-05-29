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
class IdentificationAnswer(Answer):
    entity_type: str
    name: str = None

@dataclass
class TemporalAnswer(Answer):
    date: str = None
    time: str = None
    
@dataclass
class AnalysisAnswer(Answer):
    conclusion: str

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

class IdentificationParser(TaskParser):
    def parse(self, response: str) -> IdentificationAnswer:
        type_match = re.search(r'type:\s*([^\n]+)', response, re.IGNORECASE)
        name_match = re.search(r'name:\s*([^\n]+)', response, re.IGNORECASE)
        
        if not type_match:
            raise ValueError("Missing type in structured format")
            
        entity_type = type_match.group(1).strip()
        name = name_match.group(1).strip() if name_match else None
        
        return IdentificationAnswer(entity_type=entity_type, name=name)

class TemporalParser(TaskParser):
    def parse(self, response: str) -> TemporalAnswer:
        date_match = re.search(r'date:\s*([^\n]+)', response, re.IGNORECASE)
        time_match = re.search(r'time:\s*([^\n]+)', response, re.IGNORECASE)
        
        date = date_match.group(1).strip() if date_match else None
        time = time_match.group(1).strip() if time_match else None
        
        if not date:
            raise ValueError("Missing date in structured format")
            
        return TemporalAnswer(date=date, time=time)

class AnalysisParser(TaskParser):
    def parse(self, response: str) -> AnalysisAnswer:
        conclusion_match = re.search(r'conclusion:\s*([^\n]+)', response, re.IGNORECASE)
        
        if not conclusion_match:
            raise ValueError("Missing conclusion in structured format")
            
        conclusion = conclusion_match.group(1).strip()
        
        return AnalysisAnswer(conclusion=conclusion)

def get_parser(task_type: str) -> TaskParser:
    parsers = {
        'location': LocationParser(),
        'geolocation': LocationParser(),
        'identification': IdentificationParser(),
        'person_id': IdentificationParser(),
        'object_id': IdentificationParser(),
        'temporal': TemporalParser(),
        'date': TemporalParser(),
        'time': TemporalParser(),
        'analysis': AnalysisParser(),
    }
    
    if task_type not in parsers:
        # Default to location for backward compatibility
        return LocationParser()
    
    return parsers[task_type]

def parse_response(response: str, task_type: str = "location") -> Answer:
    """Parse structured response based on task type"""
    parser = get_parser(task_type)
    return parser.parse(response)

# Simple evaluation functions
def evaluate_answer(parsed_answer: Answer, ground_truth: dict, task_type: str) -> dict:
    """Simple evaluation - returns score and whether it's correct"""
    
    if task_type in ['location', 'geolocation']:
        if isinstance(parsed_answer, LocationAnswer):
            import haversine
            distance_km = haversine.haversine(
                (parsed_answer.lat, parsed_answer.lng),
                (ground_truth['lat'], ground_truth['lng'])
            )
            
            # Simple scoring based on distance
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
    
    elif task_type in ['identification', 'person_id', 'object_id']:
        if isinstance(parsed_answer, IdentificationAnswer):
            type_correct = (ground_truth.get('type', '').lower() in parsed_answer.entity_type.lower() or
                           parsed_answer.entity_type.lower() in ground_truth.get('type', '').lower())
            
            name_correct = True
            if ground_truth.get('name') and parsed_answer.name:
                name_correct = (ground_truth['name'].lower() in parsed_answer.name.lower() or
                               parsed_answer.name.lower() in ground_truth['name'].lower())
            
            score = 1.0 if (type_correct and name_correct) else 0.0
            return {
                'score': score,
                'correct': score == 1.0,
                'type_correct': type_correct,
                'name_correct': name_correct
            }
    
    elif task_type in ['analysis']:
        # TODO: llm judge?
        pass

    # Default fallback
    return {'score': 0.5, 'correct': False}

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