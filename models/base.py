import base64
import requests
import os
from abc import ABC, abstractmethod
from ratelimit import limits, sleep_and_retry
from typing import List, Tuple

from context import get_case

from util import get_logger
logger = get_logger(__name__)

def get_image_media_type(image_path: str) -> str:
    """Determine the media type based on file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext == '.png':
        return "image/png"
    elif ext in ['.jpg', '.jpeg']:
        return "image/jpeg"
    elif ext == '.webp':
        return "image/webp"
    else:
        return "image/jpeg" # Default fallback

def encode_image(image_path: str) -> tuple[str, str]:
    """Encode image to base64 and return with media type."""
    media_type = get_image_media_type(image_path)
    with open(image_path, "rb") as img_file:
        img_data = base64.b64encode(img_file.read()).decode("utf-8")
    return img_data, media_type

class BaseMultimodalModel(ABC):
    """Abstract base class for multimodal models."""
    api_key_name: str = None
    model_identifier: str = None
    name: str = None
    rate_limit: int = 5 # RPM
    rate_limit_period: int = 60
    max_tokens: int = 128000
    temperature: float = 0.4

    def __init__(self, api_key: str):
        if not self.name:
            self.name = self.__class__.__name__
        if not self.api_key_name:
            raise NotImplementedError(f"api_key_name must be set in {self.name} or its parent provider class")
        if not self.model_identifier:
             raise NotImplementedError(f"model_identifier must be set in {self.name}")
        self.api_key = api_key

    def _build_headers(self) -> dict: 
        """Build request headers specific to the client."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    @abstractmethod
    def _build_payload(self, text_content: List[str], encoded_images: List[Tuple[str, str]]) -> dict: 
        """Build the API payload using text content and encoded images."""
        pass
    
    @abstractmethod
    def get_token_usage(self, response: requests.Response) -> dict:
        """Get the token usage for a response."""
        pass

    def _get_endpoint(self) -> str: 
        """Get the API endpoint URL."""
        return self.base_url

    def get_tools(self) -> List[str]:
        """Get the tools for the model."""
        return [tool.get('name') for tool in self.tools]

    def _execute_function_call(self, function_name: str, function_args: dict) -> dict:
        """Execute a function call and return the result."""
        logger.debug(f"Executing function call: {function_name}")
        import tools

        def convert_bytes_to_str(obj):
            if isinstance(obj, dict):
                return {k: convert_bytes_to_str(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_bytes_to_str(elem) for elem in obj]
            elif isinstance(obj, bytes):
                try:
                    return obj.decode('utf-8')
                except UnicodeDecodeError:
                    return obj.hex()
            else:
                return obj
        
        try:
            if hasattr(tools, function_name):
                function = getattr(tools, function_name)
                logger.debug(f"Found function {function_name} in tools module")
                result = function(**function_args)
            else:
                logger.error(f"Unknown function: {function_name}")
                return {"error": f"Unknown function: {function_name}"}

            # Sanitize the result to ensure it's JSON serializable
            sanitized_result = convert_bytes_to_str(result)
            logger.debug(f"Function {function_name} executed successfully")
            return {"result": sanitized_result}

        except Exception as e:
            logger.error(f"Function execution failed for {function_name}: {str(e)}")
            return {"error": f"Function execution failed: {str(e)}"}

    @abstractmethod
    def _extract_response_text(self, response: requests.Response) -> str: 
        """Extract text from the API response."""
        pass

    
    def save_json(self, response, run_folder: str, name = None) -> None:
        case = get_case()
        
        if run_folder and case.case_id:
            logger.debug(f"Saving JSON response for case {case.case_id} to {run_folder}/json/")
            os.makedirs(f"{run_folder}/json/", exist_ok=True)
            
            if name:
                name = f"{name}_{case.case_id}"
            else:
                name = case.case_id

            i = 0
            while os.path.exists(f"{run_folder}/json/{name}_{i}.json"):
                i += 1

            with open(f"{run_folder}/json/{name}_{i}.json", "w", encoding="utf-8") as f:
                import json
                if isinstance(response, requests.Response):
                    try:
                        response_json = response.json()
                        json.dump(response_json, f, indent=2, ensure_ascii=False)
                    except (json.JSONDecodeError, ValueError):
                        logger.warning(f"Response is not valid JSON, saving as text")
                        f.write(response.text)
                elif isinstance(response, (dict, list)):
                    json.dump(response, f, indent=2, ensure_ascii=False)
                elif isinstance(response, str):
                    try:
                        parsed = json.loads(response)
                        json.dump(parsed, f, indent=2, ensure_ascii=False)
                    except json.JSONDecodeError:
                        f.write(response)
                else:
                    response_str = str(response)
                    try:
                        parsed = json.loads(response_str)
                        json.dump(parsed, f, indent=2, ensure_ascii=False)
                    except json.JSONDecodeError:
                        f.write(response_str)

    def _save_conversation_state(self, run_folder: str):
        """Save current payload and response state."""
        case = get_case()
        if not (run_folder and case and case.case_id):
            return
        
        self.save_json(self.payload, run_folder, f"payload")
        self.save_json(self.response, run_folder, f"response")

    @abstractmethod
    def _is_model_finished(self, response_json: dict) -> bool:
        """Check if the model is finished."""
        logger.warning(f"Model finish check not implemented for {self.__class__.__bases__[0].__name__}")
        return True

    def query(self, prompt: str, run_folder: str = None) -> str:
        """
        Query the model.
        """
        logger.info(f"Starting query for model {self.name}")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        case = get_case()
        if case:
            logger.debug(f"Case has {len(case.images)} images")

        def api():
            if case:
                encoded_images = [encode_image(image_path) for image_path in case.images]
                logger.debug(f"Encoded {len(encoded_images)} images")
            else:
                encoded_images = []
            
            self.headers = self._build_headers()
            self.payload = self._build_payload([prompt], encoded_images)
            self.endpoint = self._get_endpoint()
            
            logger.debug(f"Making API request to {self.endpoint}")
            
            try:
                self.response = requests.post(self.endpoint, headers=self.headers, json=self.payload)
                self.response.raise_for_status()
                logger.info(f"API request successful for {self.name}")

                while not self._is_model_finished(self.response.json()):
                    logger.debug("Model has not finished, continuing conversation with function calls")
                    response_json = self.response.json()
                    self._save_conversation_state(run_folder)
                    self._handle_function_calls(response_json)

                self._save_conversation_state(run_folder)
                
                result = self._extract_response_text(self.response)
                logger.info(f"Successfully extracted response text from {self.name}")
                return result
            except requests.exceptions.RequestException as e:
                status_code = e.response.status_code if e.response is not None else "N/A"
                error_text = e.response.text if e.response is not None else str(e)
                logger.error(f"API error ({status_code}) for {self.name}: {error_text}")
                raise Exception(f"{self.name} API error ({status_code})") from e
            except Exception as e:
                logger.error(f"Unexpected error in {self.name} core logic: {str(e)}")
                raise

        call = sleep_and_retry(
            limits(calls=self.rate_limit, period=self.rate_limit_period)(api)
        )

        return call()