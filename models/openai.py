from models.base import BaseMultimodalModel, logger
from typing import List, Tuple
import requests

from tools import TOOLS_BASIC

class OpenAIClient(BaseMultimodalModel):
    api_key_name = "OPENAI_API_KEY"
    base_url = "https://api.openai.com/v1/responses"
    provider = "OpenAI"

    reasoning_effort: str = None
    detail: str = None
    tools: List = None

    def get_token_usage(self, response: requests.Response) -> dict:
        """Get the token usage for a response."""
        #TODO
        response_json = response.json()
        return {
            "input_tokens": response_json.get("usage", {}).get("input_tokens", 0),
            "output_tokens": response_json.get("usage", {}).get("output_tokens", 0)
        }

    def _build_payload(self, text_content: List[str], encoded_images: List[Tuple[str, str]] = None) -> dict:
        logger.debug(f"Building OpenAI payload with {len(text_content)} text items and {len(encoded_images) if encoded_images else 0} images")
        content = []
        for text in text_content:
            content.append({"type": "input_text", "text": text})
        
        if encoded_images:
            for img_data, media_type in encoded_images:
                image_url = f"data:{media_type};base64,{img_data}"
                image_content = {
                    "type": "input_image",
                    "image_url": image_url
                }
                if self.detail:
                    image_content["detail"] = self.detail
                content.append(image_content)

        payload = {
            "model": self.model_identifier,
            "input": [{"role": "user", "content": content}],
        }

        if hasattr(self, 'temperature') and self.temperature >= 0:
             payload["temperature"] = self.temperature

        if hasattr(self, 'max_tokens') and self.max_tokens > 0:
             payload["max_output_tokens"] = self.max_tokens

        if self.reasoning_effort:
             payload["reasoning"] = {"effort": self.reasoning_effort}
             logger.debug(f"Added reasoning effort: {self.reasoning_effort}")

        if self.tools:
             payload["tools"] = self.tools

             for tool in self.tools:
                 tool['type'] = 'function'

             
             logger.debug(f"Added {len(self.tools)} tools to OpenAI payload")
        return payload

    def _is_model_finished(self, response_json: dict) -> bool:
        """The model is finished when it returns STOP with NO function calls."""
        try:
            parts = response_json['output']
            status = response_json.get('status', '')
            
            has_function_calls = any('function_call' in part.get('type', '') for part in parts)
            
            if status == 'completed':
                is_finished = not has_function_calls
                logger.debug(f"Model finish check: status={status}, has_function_calls={has_function_calls}, finished={is_finished}")
                return is_finished
            
            # Other finish reasons indicate completion
            is_finished = status in ['cancelled', 'failed', 'incomplete']
            logger.debug(f"Model finish check: status={status}, finished={is_finished}")
            return is_finished
            
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Error checking if model is finished: {e}, assuming finished")
            return True

    def _handle_function_calls(self, response_json: dict) -> None:
        parts = response_json['output']
        
        function_calls = []
        
        for part in parts:
            if part.get('type') == 'function_call':
                function_calls.append(part)
        
        if not function_calls:
            return
        
        # Execute function calls
        if function_calls:
            logger.function_call(f"{len(function_calls)}: {function_calls}")
            
            function_responses = []
            for func_call in function_calls:
                call_id = func_call.get('call_id')
                func_name = func_call['name']
                func_args = func_call.get('arguments', '{}')

                try:
                    import json
                    func_args_dict = json.loads(func_args)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse function arguments: {func_args}")
                    func_args_dict = {}
                
                result = self._execute_function_call(func_name, func_args_dict)
                function_responses.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": str(result)
                })
                logger.debug(f"Function {func_name} completed")

        # Get the previous response ID
        previous_response_id = response_json.get('id')
        if not previous_response_id:
            logger.warning("No previous response ID found")
            return
        
        # Create a NEW payload with only the function outputs
        # This avoids repeating the entire conversation history
        follow_up_payload = {
            "previous_response_id": previous_response_id,  # This maintains context,
            "model": self.model_identifier,
            "input": function_responses,  # Only send the function outputs
            "tools": self.tools
        }
        
        # Copy over other parameters if they exist
        if hasattr(self, 'temperature') and self.temperature >= 0:
            follow_up_payload["temperature"] = self.temperature
        
        if hasattr(self, 'max_tokens') and self.max_tokens > 0:
            follow_up_payload["max_output_tokens"] = self.max_tokens
        
        if self.reasoning_effort:
            follow_up_payload["reasoning"] = {"effort": self.reasoning_effort}
            follow_up_payload["reasoning"]["summary"] = "detailed"
        
        self.payload = follow_up_payload
        
        logger.debug("Making follow-up API request with function responses")
        
        try:
            self.response = requests.post(self.endpoint, headers=self.headers, json=follow_up_payload, timeout=600)
            self.response.raise_for_status()
            logger.debug("Follow-up API request successful")
        except requests.exceptions.Timeout:
            logger.error("Follow-up API request timed out")
            raise Exception("API request timed out")
        except requests.exceptions.RequestException as e:
            logger.error(f"Follow-up API request failed: {str(e)}")
            raise

    def _extract_response_text(self, response: requests.Response) -> str:
        logger.debug("Extracting text from OpenAI response")
        response_json = response.json()
        try:
            status = response_json.get("status")
            if status != "completed":
                error_details = response_json.get("error") or response_json.get("incomplete_details") or f"Status: {status}"
                raise ValueError(f"Response generation not completed: {error_details}")
            
            response_text = ""
            for item in response_json['output']:
                if item.get('type') == "message":
                    content_items = item['content']
                
                    response_text += '\n'.join(
                        item.get('text', '')
                        for item in content_items
                        if item.get('type') == 'output_text'
                    )
                elif item.get('type') == "web_search_call":
                    response_text += f"<web_search_call>\n"

            logger.debug(f"Extracted OpenAI response text ({len(response_text)} chars)")
            return response_text
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            raise ValueError(f"Could not extract text from OpenAI v1/responses structure: {response_json}") from e
        

class GPT4o_NoTools(OpenAIClient):
    name = "GPT-4o"
    model_identifier = "gpt-4o"
    rate_limit = 4

class GPT4o_mini_NoTools(OpenAIClient):
    name = "GPT-4o-mini"
    model_identifier = "gpt-4o-mini"
    rate_limit = 6

class GPT4o(OpenAIClient):
    name = "GPT-4o"
    model_identifier = "gpt-4o"
    rate_limit = 4

    tools = [{"type": "web_search_preview"}]

class O3(OpenAIClient):
    name = "o3"
    model_identifier = "o3"
    rate_limit = 4
    max_tokens = -1
    temperature = -1

    reasoning_effort = "medium"

    tools = TOOLS_BASIC

class O3high(OpenAIClient):
    name = "o3-high"
    model_identifier = "o3"
    rate_limit = 4
    reasoning_effort = "high"
    detail = "high"
    max_tokens = -1
    temperature = -1
    tools = [{"type": "web_search_preview"}]

class O4minihigh(OpenAIClient):
    name = "o4-mini-high"
    model_identifier = "o4-mini"
    rate_limit = 5
    reasoning_effort = "high"
    max_tokens = -1
    temperature = -1
    tools = [{"type": "web_search_preview"}]