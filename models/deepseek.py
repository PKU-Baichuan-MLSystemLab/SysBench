import openai
from openai import OpenAI


class deepseek(object):
    def __init__(self) -> None:
        API_BASE = "https://api.deepseek.com"
        API_KEY = "<enter your key>"

        self.model = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE
            )
    
    def __call__(self, messages):
        response = self.model.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        return response.choices[0].message.content
    
        
if __name__ == "__main__":
    model = deepseek()    
    messages = [{"role": "system", "content": "你是百灵鸟，你是一个给人看病的医生"}, {"role": "user", "content": "你叫什么名字"}]
    print(model(messages))
    
    
    
