import sys
import json 
import random 
# import openai 
from openai import OpenAI

random.seed(1234)

class gpt4o(object):
    def __init__(self, model_name="gpt-4o", key="<enter your key>") -> None:
        self.client = OpenAI(api_key=key)
        self.model_name = model_name
        print(f"model_name: {self.model_name}")
    
    def __call__(self, query, retry=10, temperature=None):
        if isinstance(query, str):
            messages = [{"role":"user","content": query}]
        elif isinstance(query, list):
            messages = query
        else:
            raise ValueError("query must be str or list")
        i = 0
        while i < retry:
            try:
                if temperature is None:
                    response = self.client.chat.completions.create(
                        model = self.model_name,
                        messages=messages,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model = self.model_name,
                        messages=messages,
                        temperature=temperature
                    )
                   
                result = response.choices[0].message.content
                
                assert isinstance(result, str) and response.choices[0].finish_reason == "stop"
                return result
            except Exception as e:
                print(e)
        else:
            raise 

if __name__ == "__main__":
    gpt4 = gpt4o()
    messages = [{"role": "system", "content": "你是百灵鸟，你是一个给人看病的医生"}, {"role": "user", "content": "你叫什么名字"}]
    print(gpt4(messages))
    
    
