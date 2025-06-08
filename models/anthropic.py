from models.base import BaseMultimodalModel, logger, get_image_media_type
from typing import List, Tuple
import requests
import base64
import io
from PIL import Image
import os

from tools import TOOLS

class AnthropicClient(BaseMultimodalModel):
    api_key_name = "ANTHROPIC_API_KEY"
    base_url = "https://api.anthropic.com/v1/messages"
    anthropic_version: str = "2023-06-01"
    beta_header: str = None
    enable_thinking: bool = False
    tools: List = None

    def _encode_image(self, image_path: str, max_file_size: int = 4 * 1024 * 1024) -> tuple[str, str]:
        """Encode image to base64, ensuring it stays under max_file_size bytes."""
        logger.debug(f"Encoding image: {image_path} (max size: {max_file_size} bytes)")
        media_type = get_image_media_type(image_path)
        
        with Image.open(image_path) as img:
            original_size = os.path.getsize(image_path)
            logger.debug(f"Original image size: {original_size} bytes, dimensions: {img.size}")
            
            if img.mode == 'RGBA' and media_type == 'image/jpeg':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
                logger.debug("Converted RGBA to RGB for JPEG encoding")
            
            # Start with original size and high quality
            current_img = img.copy()
            quality = 95
            scale_factor = 1.0
            
            while True:
                img_buffer = io.BytesIO()
                
                if media_type == 'image/png':
                    current_img.save(img_buffer, format='PNG', optimize=True)
                else:
                    current_img.save(img_buffer, format='JPEG', quality=quality, optimize=True)
                
                img_size = img_buffer.tell()
                
                # Check if we're under the limit
                if img_size <= max_file_size:
                    if img_size != original_size:
                        logger.info(f"Downscaled image to {img_size} bytes (from {original_size} bytes)")
                    else:
                        logger.debug(f"Image already under size limit: {img_size} bytes")
                    img_data = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
                    break
                
                # Reduce quality first (for JPEG)
                if media_type == 'image/jpeg' and quality > 20:
                    quality -= 10
                    logger.debug(f"Reduced JPEG quality to {quality}")
                # Then reduce dimensions
                else:
                    scale_factor *= 0.9
                    new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
                    current_img = img.resize(new_size, Image.Resampling.LANCZOS)
                    quality = 85  # Reset quality when we resize
                    logger.debug(f"Resized image to {new_size} (scale factor: {scale_factor:.2f})")
                
                # Safety check to avoid infinite loop
                if scale_factor < 0.1:
                    logger.error(f"Cannot compress image {image_path} to under {max_file_size} bytes")
                    raise ValueError(f"Cannot compress image {image_path} to under {max_file_size} bytes")
        
        return img_data, media_type

    def _build_headers(self) -> dict:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json"
        }
        effective_beta_header = getattr(self, 'beta_header', None)
        if effective_beta_header:
             headers["anthropic-beta"] = effective_beta_header
             logger.debug(f"Added beta header: {effective_beta_header}")
        return headers

    def _build_payload(self, text_content: List[str], encoded_images: List[Tuple[str, str]] = None) -> dict:
        logger.debug(f"Building Anthropic payload with {len(text_content)} text items and {len(encoded_images) if encoded_images else 0} images")
        content = []
        # Text
        for text in text_content:
            content.append({"type": "text", "text": text})
        
        # Images
        if encoded_images:
            for img_data, media_type in encoded_images:
                content.append({
                    "type": "image", 
                    "source": {
                        "type": "base64", 
                        "media_type": media_type, 
                        "data": img_data
                    }
                })

        payload = {
            "model": self.model_identifier,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": content}]
        }

        if self.tools:
            for tool in self.tools:
                tool["input_schema"] = tool["parameters"]
                del tool["parameters"]

            payload["tools"] = self.tools
            logger.debug(f"Added {len(self.tools)} tools to payload")
        
        if self.enable_thinking:
            payload["thinking"] = {"type": "enabled", "budget_tokens": self.max_tokens - 32000}
            if "temperature" in payload:
                del payload["temperature"]
            logger.debug("Enabled thinking mode")
        return payload

    def _handle_function_calls(self, response_json: dict) -> str:
        parts = response_json['content']

        function_calls = []
        text_parts = []
        thinking_parts = []
        
        for part in parts:
            if part.get('type') == 'tool_use':
                function_calls.append(part)
            elif part.get('type') == 'text':
                text_parts.append(part['text'])
            elif part.get('type') == 'thinking':
                thinking_parts.append(part)

        # If there are function calls, execute them and continue the conversation
        if function_calls:
            logger.function_call(f"{len(function_calls)}: {function_calls}")
            
            # Execute all function calls
            function_responses = []
            for func_call in function_calls:
                func_id = func_call['id']
                func_name = func_call['name']
                func_args = func_call.get('input', {})
                logger.debug(f"Calling {func_name} with args: {func_args}")
                
                result = self._execute_function_call(func_name, func_args)
                function_responses.append({
                    "type": "tool_result",
                    "tool_use_id": func_id,
                    "content": str(result)
                })
                logger.debug(f"Function {func_name} completed")
            
            # Build assistant message content: thinking blocks FIRST, then tool_use blocks
            assistant_content = thinking_parts + function_calls  # ✅ Thinking first!
            
            # Add model's function call response to the existing payload
            self.payload["messages"].append({
                "content": assistant_content,
                "role": "assistant"
            })
            
            # Add function responses to the existing payload
            self.payload["messages"].append({
                "content": function_responses, 
                "role": "user"
            })
            
            # Make follow-up request with the updated payload
            logger.debug("Making follow-up API request with function responses")
            headers = self._build_headers()
            endpoint = self._get_endpoint()
            
            try:
                self.response = requests.post(endpoint, headers=headers, json=self.payload, timeout=600)
                self.response.raise_for_status()
                logger.debug("Follow-up API request successful")
            except requests.exceptions.Timeout:
                logger.error("Follow-up API request timed out")
                raise Exception("API request timed out")
            except requests.exceptions.RequestException as e:
                logger.error(f"Follow-up API request failed: {str(e)}")
                raise
            
            # Extract final text response
            final_parts = self.response.json()['content']
            return ''.join(part.get('text', '') for part in final_parts)
        
        # No function calls, just return text
        logger.debug("No function calls found, returning text parts")
        return ''.join(text_parts)

    def _is_model_finished(self, response_json: dict) -> bool:
        """The model is finished when it returns STOP with NO function calls."""
        try:
            finish_reason = response_json.get('stop_reason', '')
            has_function_calls = False

            parts = response_json.get('content', [])
            for part in parts:
                if part.get('type') == 'tool_use' or part.get('type') == 'server_tool_use' or part.get('type') == 'pause_turn':
                    has_function_calls = True
                    break
            
            # Only finished if STOP with no function calls
            if finish_reason == 'end_turn':
                is_finished = not has_function_calls
                logger.debug(f"Model finish check: reason={finish_reason}, has_function_calls={has_function_calls}, finished={is_finished}")
                return is_finished
            
            # Other finish reasons indicate completion
            is_finished = finish_reason in ['refusal', 'stop_sequence', 'max_token']
            logger.debug(f"Model finish check: reason={finish_reason}, finished={is_finished}")
            return is_finished
            
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Error checking if model is finished: {e}, assuming finished")
            return True

    def _extract_response_text(self, response: requests.Response) -> str:
        logger.debug("Extracting text from Anthropic response")
        response_json = response.json()
        if not response_json.get("content"):
             raise ValueError(f"Unexpected Anthropic response format: {response_json}")
        thinking_text, response_text = "", ""
        for block in response_json["content"]:
            if block.get("type") == "thinking": thinking_text = block.get("thinking", "")
            elif block.get("type") == "text": response_text = block.get("text", "")

        if self.enable_thinking and thinking_text:
             logger.debug(f"Response includes thinking text ({len(thinking_text)} chars)")
             return f"<thinking>{thinking_text}</thinking>\n\n{response_text}"
        elif response_text:
             logger.debug(f"Extracted response text ({len(response_text)} chars)")
             return response_text
        else:
             raise ValueError(f"Could not extract text from Anthropic response: {response_json}")
        
class Claude3_7Sonnet(AnthropicClient):
    name = "Claude 3.7 Sonnet"
    model_identifier = "claude-3-7-sonnet-20250219"
    
    tools = [{"type": "web_search_20250305", "name": "web_search"}]
class Claude3_7SonnetThinking(AnthropicClient):
    name = "Claude 3.7 Sonnet (Thinking)"
    model_identifier = "claude-3-7-sonnet-20250219"
    enable_thinking = True
    rate_limit = 2
    beta_header = "output-128k-2025-02-19"

    tools = [{"type": "web_search_20250305", "name": "web_search"}]
class Claude4SonnetThinking(AnthropicClient):
    name = "Claude 4 Sonnet (Thinking)"
    model_identifier = "claude-sonnet-4-20250514"
    enable_thinking = True
    rate_limit = 0.2
    beta_header = "output-128k-2025-02-19"

    # tools = [{"type": "web_search_20250305", "name": "web_search"}]
    tools = TOOLS
class Claude4OpusThinking(AnthropicClient):
    name = "Claude 4 Opus (Thinking)"
    model_identifier = "claude-opus-4-20250514"
    enable_thinking = True
    rate_limit = 0.2
    beta_header = "output-128k-2025-02-19"

    tools = [{"type": "web_search_20250305", "name": "web_search"}]
