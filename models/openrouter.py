from models.base import BaseMultimodalModel, logger
from typing import List, Tuple
import requests

from tools import TOOLS_BASIC


class OpenRouterClient(BaseMultimodalModel):
    api_key_name = "OPENROUTER_API_KEY"
    base_url = "https://openrouter.ai/api/v1/chat/completions"
    referer_url: str = "https://osintbench.org"

    tools: List = None

    def get_token_usage(self, response: requests.Response) -> dict:
        """Get the token usage for a response."""
        response_json = response.json()
        return {
            "input_tokens": response_json.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": response_json.get("usage", {}).get("completion_tokens", 0)
        }

    def _build_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.referer_url
        }

    def _build_payload(self, text_content: List[str], encoded_images: List[Tuple[str, str]] = None) -> dict:
        logger.debug(f"Building OpenRouter payload with {len(text_content)} text items and {len(encoded_images) if encoded_images else 0} images")
        content = []
        for text in text_content:
            content.append({"type": "text", "text": text})
        
        if encoded_images:
            for img_data, media_type in encoded_images:
                image_url = f"data:{media_type};base64,{img_data}"
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
                if hasattr(self, 'detail') and self.detail:
                    image_content["image_url"]["detail"] = self.detail
                content.append(image_content)

        payload = {
            "model": self.model_identifier,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature
        }

        # Add tools if available
        if self.tools:
            # Convert tools to OpenRouter format (OpenAI-compatible)
            openrouter_tools = []
            for tool in self.tools:
                openrouter_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                }
                openrouter_tools.append(openrouter_tool)
            
            payload["tools"] = openrouter_tools
            logger.debug(f"Added {len(openrouter_tools)} tools to OpenRouter payload")

        return payload

    def _is_model_finished(self, response_json: dict) -> bool:
        """The model is finished when it returns without function calls or with finish_reason 'stop'."""
        try:
            if not response_json.get("choices"):
                logger.warning("No choices in response, assuming finished")
                return True
            
            choice = response_json["choices"][0]
            finish_reason = choice.get("finish_reason", "")
            
            # Check if there are function calls
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls", [])
            
            if tool_calls:
                logger.debug(f"Model has {len(tool_calls)} tool calls, not finished")
                return False
            
            # Model is finished if finish_reason is 'stop' or other completion reasons
            is_finished = finish_reason in ['stop', 'length', 'content_filter', 'null']
            logger.debug(f"Model finish check: finish_reason={finish_reason}, finished={is_finished}")
            return is_finished
            
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Error checking if model is finished: {e}, assuming finished")
            return True

    def _handle_function_calls(self, response_json: dict) -> None:
        """Handle function calls in OpenRouter response format."""
        if not response_json.get("choices"):
            return
        
        choice = response_json["choices"][0]
        message = choice.get("message", {})
        tool_calls = message.get("tool_calls", [])
        
        if not tool_calls:
            return
        
        logger.function_call(f"{len(tool_calls)}: {tool_calls}")
        
        # Execute function calls
        function_responses = []
        for tool_call in tool_calls:
            call_id = tool_call.get("id")
            function_data = tool_call.get("function", {})
            func_name = function_data.get("name")
            func_args = function_data.get("arguments", "{}")
            
            try:
                import json
                func_args_dict = json.loads(func_args)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse function arguments: {func_args}")
                func_args_dict = {}
            
            result = self._execute_function_call(func_name, func_args_dict)
            function_responses.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": func_name,
                "content": str(result.get("result", result))
            })
            logger.debug(f"Function {func_name} completed")
        
        # Build new payload with conversation history
        # First, get current messages and add the assistant's response with tool calls
        current_messages = self.payload.get("messages", [])
        
        # Add the assistant's message with tool calls
        assistant_message = {
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": tool_calls
        }
        current_messages.append(assistant_message)
        
        # Add the function responses
        current_messages.extend(function_responses)
        
        # Create follow-up payload
        follow_up_payload = {
            "model": self.model_identifier,
            "messages": current_messages,
            "tools": self.payload.get("tools", [])  # Keep the same tools
        }
        
        # Copy over other parameters if they exist
        if "temperature" in self.payload:
            follow_up_payload["temperature"] = self.payload["temperature"]
        
        if "max_tokens" in self.payload:
            follow_up_payload["max_tokens"] = self.payload["max_tokens"]
        
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
        logger.debug("Extracting text from OpenRouter response")
        response_json = response.json()
        try:
            message = response_json["choices"][0]["message"]
            content = message.get("content", "")
            
            # Handle the case where content is None (e.g., when there are only tool calls)
            if content is None:
                content = ""
            
            # If there are tool calls but no content, indicate that functions were called
            tool_calls = message.get("tool_calls", [])
            if tool_calls and not content:
                function_names = [tc.get("function", {}).get("name", "unknown") for tc in tool_calls]
                content = f"[Called functions: {', '.join(function_names)}]"
            
            logger.debug(f"Extracted OpenRouter response text ({len(content)} chars)")
            return content
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Could not extract text from OpenRouter response: {response_json}") from e


class HorizonBeta(OpenRouterClient):
    name = "Horizon Beta"
    provider = "OpenAI"
    model_identifier = "openrouter/horizon-beta"
    rate_limit = 2
    
    tools = TOOLS_BASIC