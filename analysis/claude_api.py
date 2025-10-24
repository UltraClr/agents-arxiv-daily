import os
import sys
import logging
from openai import OpenAI

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class OpenAIClient:
    """OpenAI API client for chat completions."""

    def __init__(self, api_key, base_url="https://api.openai.com", model_name="claude-sonnet-4-5-20250929"):
        """
        Initialize OpenAI client
        @param api_key: OpenAI API key
        @param default_url: Base URL for API endpoint
        @param model_name: Model to use
        """
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=120.0,
            max_retries=3
        )
        logging.info(f"OpenAI client initialized with model: {model_name}")

    def send_message(self, prompt_content):
        """
        Send a message to OpenAI API
        @param prompt_content: The prompt/message to send
        @return: Response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt_content,
                    },
                ],
            )

            return response.choices[0].message.content

        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            raise


if __name__ == '__main__':
    """Test OpenAI API client"""

    # Check if API key file exists
    api_key = os.environ.get('ANTHROPIC_AUTH_TOKEN')
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