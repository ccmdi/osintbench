from models.base import BaseMultimodalModel, logger
from typing import List, Tuple
import requests

from tools import TOOLS_BASIC, TOOLS_BASIC_FULL

class GoogleClient(BaseMultimodalModel):
    api_key_name = "GEMINI_API_KEY"
    base_url = "https://generativelanguage.googleapis.com"
    api_version_path: str = "" # e.g., "beta/" for experimental versions
    tools: List[str] = None
    provider = "Google"

    def _get_endpoint(self) -> str:
        action = "generateContent"
        version_path = getattr(self, 'api_version_path', '')
        return f"{self.base_url}/{version_path}/models/{self.model_identifier}:{action}?key={self.api_key}"

    def get_token_usage(self, response: requests.Response) -> dict:
        """Get the token usage for a response."""
        response_json = response.json()
        return {
            "input_tokens": response_json.get("usageMetadata", {}).get("promptTokenCount", 0),
            "output_tokens": response_json.get("usageMetadata", {}).get("candidatesTokenCount", 0)
        }

    def get_tools(self) -> List[str]:
        """Get the tools for the model."""
        if not self.tools:
            return []
        
        for tool_group in self.tools:
            if 'function_declarations' in tool_group:
                return [tool.get('name') for tool in tool_group['function_declarations']]
            return list(tool_group.keys())
        
        return []

    def _build_headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def _build_payload(self, text_content: List[str], encoded_images: List[Tuple[str, str]] = None) -> dict:
        logger.debug(f"Building Google payload with {len(text_content)} text items and {len(encoded_images) if encoded_images else 0} images")
        parts = []
        for text in text_content:
            parts.append({"text": text})
        
        if encoded_images:
            for img_data, media_type in encoded_images:
                parts.append({
                    "inline_data": {
                        "mime_type": media_type, 
                        "data": img_data
                    }
                })

        payload = {
            "contents": [{"parts": parts, "role": "user"}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens
            }
        }

        if self.tools:
            payload["tools"] = self.tools
            logger.debug(f"Added tools to Google payload: {self.tools}")

        return payload

    def _handle_function_calls(self, response_json: dict) -> None:
        logger.debug(response_json)
        parts = response_json['candidates'][0]['content']['parts']
        
        function_calls = []
        text_parts = []
        
        for part in parts:
            if 'functionCall' in part:
                function_calls.append(part['functionCall'])
            elif 'text' in part:
                text_parts.append(part['text'])

        if function_calls:
            logger.function_call(f"{len(function_calls)}: {function_calls}")
            
            # Execute all function calls
            function_responses = []
            for func_call in function_calls:
                func_name = func_call['name']
                func_args = func_call.get('args', {})
                logger.debug(f"Calling {func_name} with args: {func_args}")
                
                result = self._execute_function_call(func_name, func_args)
                function_responses.append({
                    "functionResponse": {
                        "name": func_name,
                        "response": result
                    }
                })
                logger.debug(f"Function {func_name} completed")
            
            self.payload["contents"].append({
                "parts": [{"text": part} for part in text_parts] + [{"functionCall": fc} for fc in function_calls], 
                "role": "model"
            })
            
            self.payload["contents"].append({
                "parts": function_responses, 
                "role": "user"
            })
            
            logger.debug("Making follow-up API request with function responses")

            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    if attempt > 0:
                        logger.info(f"Retrying follow-up request (attempt {attempt + 1}/{max_attempts})")
                    
                    self.response = requests.post(self.endpoint, headers=self.headers, json=self.payload, timeout=(5, 300))
                    self.response.raise_for_status()
                    logger.debug("Follow-up API request successful")
                    break
                    
                except requests.exceptions.Timeout:
                    if attempt < max_attempts - 1:
                        logger.warning(f"Follow-up API request timed out, will retry...")
                        continue  # Try again
                    else:
                        logger.error(f"Follow-up API request timed out after {max_attempts} attempts")
                        raise Exception(f"API request timed out after {max_attempts} attempts")
                        
                except requests.exceptions.RequestException as e:
                    logger.error(f"Follow-up API request failed: {str(e)}")
                    raise
    
    def _is_model_finished(self, response_json: dict) -> bool:
        """The model is finished when it returns STOP with NO function calls."""
        try:
            candidate = response_json['candidates'][0]
            parts = candidate['content']['parts']
            finish_reason = candidate.get('finishReason', '')
            
            has_function_calls = any('functionCall' in part for part in parts)
            
            # Only finished if STOP with no function calls
            if finish_reason == 'STOP':
                is_finished = not has_function_calls
                logger.debug(f"Model finish check: reason={finish_reason}, has_function_calls={has_function_calls}, finished={is_finished}")
                return is_finished
            
            # Other finish reasons indicate completion
            is_finished = finish_reason in ['MAX_TOKENS', 'SAFETY', 'RECITATION']
            logger.debug(f"Model finish check: reason={finish_reason}, finished={is_finished}")
            return is_finished
            
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Error checking if model is finished: {e}, assuming finished")
            return True

    def _extract_response_text(self, response: requests.Response) -> str:
        logger.debug("Extracting text from Google response")
        response_json = response.json()
        try:
            parts = response_json['candidates'][0]['content']['parts']
            result = ''.join(part.get('text', '') for part in parts)
            logger.debug(f"Extracted Google response text ({len(result)} chars)")
            return result
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Could not extract text from Google response: {response_json}") from e



class Gemini2Flash(GoogleClient):
    name = "Gemini 2.0 Flash"
    model_identifier = "gemini-2.0-flash"
    rate_limit = 10
    api_version_path = "v1beta"

    # tools = [{"google_search": {}}]
    tools = [{"function_declarations": TOOLS_BASIC}]
class Gemini2_5Pro(GoogleClient):
    name = "Gemini 2.5 Pro"
    model_identifier = "gemini-2.5-pro-preview-05-06"
    rate_limit = 2
    api_version_path = "v1beta"

    tools = [{"function_declarations": TOOLS_BASIC}]
class Gemini2_5Pro0605(GoogleClient):
    name = "Gemini 2.5 Pro 06-05"
    model_identifier = "gemini-2.5-pro-preview-06-05"
    rate_limit = 2
    api_version_path = "v1beta"
    
    tools = [{"function_declarations": TOOLS_BASIC}]
class Gemini2_5Flash(GoogleClient):
    name = "Gemini 2.5 Flash Preview"
    model_identifier = "gemini-2.5-flash-preview-04-17"
    rate_limit = 2
    api_version_path = "v1beta"

    tools = [{"function_declarations": TOOLS_BASIC}]