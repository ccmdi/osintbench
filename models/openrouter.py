from models.base import BaseMultimodalModel, logger
from typing import List, Tuple
import requests

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
        logger.debug(f"Building OpenRouter payload with {len(text_content)} text items and {len(encoded_images) if encoded_images else 0} images")
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
        logger.debug("Extracting text from OpenRouter response")
        response_json = response.json()
        try:
            result = response_json["choices"][0]["message"]["content"]
            logger.debug(f"Extracted OpenRouter response text ({len(result)} chars)")
            return result
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Could not extract text from OpenRouter response: {response_json}") from e