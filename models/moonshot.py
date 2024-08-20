import requests
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI


class moonshot():
    def __call__(self, messages):
        
        client = OpenAI(
            api_key = "<enter your key>",
            base_url = "https://api.moonshot.cn/v1",
        )
        Max_Try = 10
        i = 0
        response = ""
        while i < Max_Try:
            try:
                completion = client.chat.completions.create(
                    model = "moonshot-v1-8k",
                    messages = messages)
                response = completion.choices[0].message.content
                return response
            except Exception as e:
                print(f"Try {i}/{Max_Try}【response】:{response}\t message:【Error】:{e}", flush=True)
                i += 1
                continue
        return response
    

if __name__ == "__main__":
    model = moonshot()
    print(model([{"role": "system", "content": "你是百灵鸟，你是一个给人看病的医生"}, {"role": "user", "content": "你叫什么名字"}]))