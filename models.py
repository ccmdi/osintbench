import base64
import requests
import os
from abc import ABC, abstractmethod
from ratelimit import limits, sleep_and_retry
from typing import List, Tuple
from PIL import Image
import io

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

class BaseMultimodalModel(ABC):
    """Abstract base class for multimodal models."""
    api_key_name: str = None
    model_identifier: str = None
    name: str = None
    rate_limit: int = 5
    rate_limit_period: int = 60
    max_tokens: int = 64000
    temperature: float = 0.4

    def __init__(self, api_key: str):
        if not self.name:
            self.name = self.__class__.__name__
        if not self.api_key_name:
            raise NotImplementedError(f"api_key_name must be set in {self.name} or its parent provider class")
        if not self.model_identifier:
             raise NotImplementedError(f"model_identifier must be set in {self.name}")
        self.api_key = api_key

    def _encode_image(self, image_path: str) -> tuple[str, str]:
        """Encode image to base64 and return with media type."""
        media_type = get_image_media_type(image_path)
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")
        return img_data, media_type

    def _encode_case_images(self, case) -> List[Tuple[str, str]]:
        """Encode all images from a case."""
        return [self._encode_image(image_path) for image_path in case.images]

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

    def _get_endpoint(self) -> str: 
        """Get the API endpoint URL."""
        return self.base_url

    @abstractmethod
    def _extract_response_text(self, response: requests.Response) -> str: 
        """Extract text from the API response."""
        pass
        
    def query(self, prompt: str, case = None, run_folder: str = None) -> str:
        """
        Public method to query the model.
        """

        def api():
            if case:
                encoded_images = self._encode_case_images(case)
            else:
                encoded_images = []
            
            headers = self._build_headers()
            payload = self._build_payload(prompt, encoded_images)
            endpoint = self._get_endpoint()
            
            try:
                response = requests.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                if run_folder and case.case_id:
                    os.makedirs(f"{run_folder}/json/", exist_ok=True)
                    with open(f"{run_folder}/json/{case.case_id}.json", "w", encoding="utf-8") as f:
                        f.write(response.text)
                return self._extract_response_text(response)
            except requests.exceptions.RequestException as e:
                status_code = e.response.status_code if e.response is not None else "N/A"
                error_text = e.response.text if e.response is not None else str(e)
                print(f"API error ({status_code}) for {self.name}: {error_text}...")
                raise Exception(f"{self.name} API error ({status_code})") from e
            except Exception as e:
                print(f"Unexpected error in {self.name} core logic: {str(e)}")
                raise

        call = sleep_and_retry(
            limits(calls=self.rate_limit, period=self.rate_limit_period)(api)
        )

        return call()

class AnthropicClient(BaseMultimodalModel):
    api_key_name = "ANTHROPIC_API_KEY"
    base_url = "https://api.anthropic.com/v1/messages"
    anthropic_version: str = "2023-06-01"
    beta_header: str = None
    enable_thinking: bool = False
    tools: List = None

    def _encode_image(self, image_path: str, max_file_size: int = 5 * 1024 * 1024) -> tuple[str, str]:
        """Encode image to base64, ensuring it stays under max_file_size bytes."""
        media_type = get_image_media_type(image_path)
        
        with Image.open(image_path) as img:
            if img.mode == 'RGBA' and media_type == 'image/jpeg':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            
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
                    print(f"Downscaled image to {img_size} bytes")
                    img_data = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
                    break
                
                # Reduce quality first (for JPEG)
                if media_type == 'image/jpeg' and quality > 20:
                    quality -= 10
                # Then reduce dimensions
                else:
                    scale_factor *= 0.9
                    new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
                    current_img = img.resize(new_size, Image.Resampling.LANCZOS)
                    quality = 85  # Reset quality when we resize
                
                # Safety check to avoid infinite loop
                if scale_factor < 0.1:
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
        return headers

    def _build_payload(self, text_content: List[str], encoded_images: List[Tuple[str, str]] = None) -> dict:
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
            payload["tools"] = self.tools
        
        if self.enable_thinking:
            payload["thinking"] = {"type": "enabled", "budget_tokens": self.max_tokens - 32000}
            if "temperature" in payload:
                del payload["temperature"]
        return payload

    def _extract_response_text(self, response: requests.Response) -> str:
        response_json = response.json()
        if not response_json.get("content"):
             raise ValueError(f"Unexpected Anthropic response format: {response_json}")
        thinking_text, response_text = "", ""
        for block in response_json["content"]:
            if block.get("type") == "thinking": thinking_text = block.get("thinking", "")
            elif block.get("type") == "text": response_text = block.get("text", "")

        if self.enable_thinking and thinking_text:
             return f"<thinking>{thinking_text}</thinking>\n\n{response_text}"
        elif response_text:
             return response_text
        else:
             raise ValueError(f"Could not extract text from Anthropic response: {response_json}")


class GoogleClient(BaseMultimodalModel):
    api_key_name = "GEMINI_API_KEY"
    base_url = "https://generativelanguage.googleapis.com"
    api_version_path: str = "" # e.g., "beta/" for experimental versions
    tools: str = None

    def _get_endpoint(self) -> str:
        action = "generateContent"
        version_path = getattr(self, 'api_version_path', '')
        return f"{self.base_url}/{version_path}/models/{self.model_identifier}:{action}?key={self.api_key}"

    def _build_headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def _build_payload(self, text_content: List[str], encoded_images: List[Tuple[str, str]] = None) -> dict:
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
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens
            }
        }

        if self.tools:
            payload["tools"] = self.tools

        return payload

    def _extract_response_text(self, response: requests.Response) -> str:
        response_json = response.json()
        try:
            parts = response_json['candidates'][0]['content']['parts']
            return ''.join(part.get('text', '') for part in parts)
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Could not extract text from Google response: {response_json}") from e


class OpenAIClient(BaseMultimodalModel):
    api_key_name = "OPENAI_API_KEY"
    base_url = "https://api.openai.com/v1/responses"

    reasoning_effort: str = None
    detail: str = None
    tools: List = None

    def _build_payload(self, text_content: List[str], encoded_images: List[Tuple[str, str]] = None) -> dict:
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

        if self.tools:
             payload["tools"] = self.tools
        return payload

    def _extract_response_text(self, response: requests.Response) -> str:
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

            return response_text
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            raise ValueError(f"Could not extract text from OpenAI v1/responses structure: {response_json}") from e

class OpenRouterClient(BaseMultimodalModel):
    api_key_name = "OPENROUTER_API_KEY"
    base_url = "https://openrouter.ai/api/v1/chat/completions"
    referer_url: str = "https://geobench.org"

    def _build_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.referer_url
        }

    def _build_payload(self, text_content: List[str], encoded_images: List[Tuple[str, str]] = None) -> dict:
        content = []
        for text in text_content:
            content.append({"type": "text", "text": text})
        
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
        
        return {
            "model": self.model_identifier,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature
        }

    def _extract_response_text(self, response: requests.Response) -> str:
        response_json = response.json()
        try:
            return response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Could not extract text from OpenRouter response: {response_json}") from e


# Anthropic Models
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
    rate_limit = 2
    beta_header = "output-128k-2025-02-19"

    tools = [{"type": "web_search_20250305", "name": "web_search"}]
class Claude4OpusThinking(AnthropicClient):
    name = "Claude 4 Opus (Thinking)"
    model_identifier = "claude-opus-4-20250514"
    enable_thinking = True
    rate_limit = 2
    beta_header = "output-128k-2025-02-19"

    tools = [{"type": "web_search_20250305", "name": "web_search"}]


# Google Models
class Gemini2Flash(GoogleClient):
    name = "Gemini 2.0 Flash"
    model_identifier = "gemini-2.0-flash"
    rate_limit = 10
    api_version_path = "v1beta"

    tools = [{"google_search": {}}]
class Gemini2_5Pro(GoogleClient):
    name = "Gemini 2.5 Pro"
    model_identifier = "gemini-2.5-pro-preview-05-06"
    rate_limit = 2
    api_version_path = "v1beta"
    
    tools = [{"google_search": {}}]
class Gemini2_5Flash(GoogleClient):
    name = "Gemini 2.5 Flash Preview"
    model_identifier = "gemini-2.5-flash-preview-04-17"
    rate_limit = 2
    api_version_path = "v1beta"

    tools = [{"google_search": {}}]

# OpenAI Models
class GPT4o(OpenAIClient):
    name = "GPT-4o"
    model_identifier = "gpt-4o"
    rate_limit = 3
    
    tools = [{"type": "web_search_preview"}]
class O3high(OpenAIClient):
    name = "o3-high"
    model_identifier = "o3"
    rate_limit = 2
    reasoning_effort = "high"
    detail = "high"

    # NOT SUPPORTED
    # max_tokens = -1
    # temperature = -1

    
    #TODO: add tools
class O4minihigh(OpenAIClient):
    name = "o4-mini-high"
    model_identifier = "o4-mini"
    rate_limit = 5
    reasoning_effort = "high"

    # NOT SUPPORTED
    max_tokens = -1
    temperature = -1

    #TODO: add tools
