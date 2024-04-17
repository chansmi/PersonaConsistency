import os
from dotenv import load_dotenv
import openai
import google.generativeai as genai
#from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables from .env file
load_dotenv()

class LanguageModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate(self, prompt):
        raise NotImplementedError

class OpenAIBackend(LanguageModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        openai.api_key = os.getenv('OPENAI_API_KEY')

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response['choices'][0]['message']['content'].strip()

        except openai.error.OpenAIError as e:
            print(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise
'''
class AnthropicBackend(LanguageModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        Claude_API = os.getenv('CLAUDE_API_KEY')
        self.anthropic = Anthropic(api_key=Claude_API)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, prompt):
        try:
            response = self.anthropic.completions.create(
                prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
                model=self.model_name,
                max_tokens_to_sample=1000
            )
            return response.completion.strip()
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise
'''
class GeminiBackend(LanguageModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        GEMINI_API = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=GEMINI_API)
        
        generation_config = {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_output_tokens": 1000,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", 
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"  
            }
        ]
        
        self.gemini = genai.GenerativeModel(model_name=model_name,
                                            generation_config=generation_config,
                                            safety_settings=safety_settings)

    def generate(self, prompt):
        response = self.gemini.generate_content(
            contents=[
                {"role": "user", "parts": [prompt]},  
            ]
        )
        return response.text