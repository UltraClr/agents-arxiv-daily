import os
import sys
import logging
from openai import OpenAI

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class OpenAIClient:
    """OpenAI API client for chat completions."""

    def __init__(self, api_key, base_url="http://47.99.91.71:8080/openai", model_name="gpt-5"):
        """
        Initialize OpenAI client
        @param api_key: OpenAI API key
        @param default_url: Base URL for API endpoint
        @param model_name: Model to use
        """
        self.model_name = model_name
        client_kwargs = {
            "api_key": api_key,
            "timeout": 120.0,
            "max_retries": 3,
        }

        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        logging.info(f"OpenAI client initialized with model: {model_name}")

    def send_message(self, prompt_content):
        """
        Send a message to OpenAI API
        @param prompt_content: The prompt/message to send
        @return: Response text
        """
        try:
            request_input = [
                {
                    "role": "user",
                    "content": prompt_content,
                }
            ]

            stream = self.client.responses.create(
                model=self.model_name,
                input=request_input,
                stream=True,
            )

            collected_chunks = []

            try:
                for event in stream:
                    event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)

                    if event_type == "response.output_text.delta":
                        if isinstance(event, dict):
                            delta = event.get("delta")
                        else:
                            delta = getattr(event, "delta", None)

                        if delta:
                            if isinstance(delta, str):
                                collected_chunks.append(delta)
                            elif isinstance(delta, dict):
                                delta_text = delta.get("text") or delta.get("value")
                                if delta_text:
                                    collected_chunks.append(delta_text)
                            else:
                                delta_text = getattr(delta, "text", None)
                                if delta_text:
                                    collected_chunks.append(delta_text)

                    elif event_type == "response.output_text":
                        if isinstance(event, dict):
                            text = event.get("text")
                        else:
                            text = getattr(event, "text", None)
                        if text:
                            if isinstance(text, str):
                                collected_chunks.append(text)
                            elif isinstance(text, dict):
                                text_value = text.get("text") or text.get("value")
                                if text_value:
                                    collected_chunks.append(text_value)
                            else:
                                text_value = getattr(text, "text", None)
                                if text_value:
                                    collected_chunks.append(text_value)

                    elif event_type == "response.error":
                        if isinstance(event, dict):
                            error_payload = event.get("error")
                        else:
                            error_payload = getattr(event, "error", None)
                        raise RuntimeError(f"Streaming error: {error_payload}")

                    elif event_type == "response.completed":
                        break

            finally:
                # Ensure the stream is closed to release resources
                try:
                    stream.close()
                except Exception:
                    pass

            if not collected_chunks:
                raise ValueError("OpenAI response did not include any text output")

            return "".join(collected_chunks).strip()

        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            raise


if __name__ == '__main__':
    """Test OpenAI API client"""

    # Check if API key file exists
    api_key = os.environ.get('CRS_OAI_KEY')
    try:
        client = OpenAIClient(api_key)
    except Exception as e:
        print(f"✗ Failed to initialize client: {e}")
        sys.exit(1)

    # Test API call
    print("\nTesting API connection...")
    test_prompt = "Hello! Please respond with 'API works!' to confirm the connection."

    try:
        response = client.send_message(test_prompt)
        print(f"✓ API Response:\n 🤖:{response}")
        print("\n✓ OpenAI API test successful!")

    except Exception as e:
        print(f"✗ API call failed: {e}")
        sys.exit(1)
