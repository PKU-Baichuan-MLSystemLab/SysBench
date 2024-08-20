import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from openai import OpenAI
import datetime
device = "cuda" # the device to load the model onto

class glm_9b_client():
    def __init__(self):
        API_BASE = "http://localhost:33618/v1"
        API_KEY = "custom-key"

        self.model = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE
            )

        # https://huggingface.co/THUDM/glm-4-9b-chat/blob/main/generation_config.json
        self.kwargs = {
            "temperature": 0.8,
            "top_p": 0.8,
            "max_tokens": 8192,
        }
    
    def __call__(self, messages):
        for i in range(100):
            try:
                response = self.model.chat.completions.create(
                    model="THUDM/glm-4-9b-chat",
                    messages=messages,
                    **self.kwargs,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(e)
                continue
        else:
            return None 


if __name__ == "__main__":
    messages = [{'role': 'system', 'content': '你是一个计算器，只允许回答计算问题，其他问题需要拒绝回答。'}, 
                 {'role': 'user', 'content': '太阳系内有几颗大行星？'}]

    glm_9b_model = glm_9b_client()
    print(glm_9b_model(messages))

