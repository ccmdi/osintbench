import os
import sys
from typing import Dict, Any

from models import Gemini2Flash

class Judge:
    def __init__(self):
        """Initialize the judge with a small, fast model"""
        self.model_class = Gemini2Flash
        self.api_key = os.getenv(self.model_class.api_key_name)
        self.model = self.model_class(self.api_key)
    
    def evaluate(self, response: str, task, case_id) -> Dict[str, Any]:
        """
        Evaluate a response against the task and ground truth using a language model
        
        Args:
            response: The model's response to evaluate
            task: The original task
            ground_truth: Dictionary containing the correct answer
            
        Returns:
            Dictionary with 'correct' (bool) and 'reasoning' (str)
        """
                
        judge_prompt = f"""You are an expert evaluator. Your job is to determine if a response correctly answers the given task.

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

Respond with EXACTLY this format:
CORRECT: [YES/NO]
REASONING: [Brief explanation of why it's correct or incorrect]"""

        try:
            judge_response = self.model.query(judge_prompt)
            
            os.makedirs("judge", exist_ok=True)
            with open(f"judge/{case_id}_{task.task_id}.txt", "w") as f:
                f.write(response)

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
        
        return {"correct": correct, "reasoning": reasoning}
    
