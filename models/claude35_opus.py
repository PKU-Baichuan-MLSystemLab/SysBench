import openai
from openai import OpenAI


class claude35_opus(object):
    def __init__(self) -> None:
        API_BASE = "http://<enter ip>/v1"
        API_KEY = "<enter your key>"

        self.model = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE
        )
    
    def __call__(self, messages):
        for i in range(100):
            try:
                response = self.model.chat.completions.create(
                    model="claude-3-opus-20240229",
                    messages=messages
                )
                print(response.json())
                return response.choices[0].message.content
            except Exception as e:
                print(e)
                continue
        else:
            return None 
        
if __name__ == "__main__":
    model = claude35_opus()    
    messages = [{"role": "system", "content": "你的名字叫百灵鸟，你擅长给人看病"}, {"role": "user", "content": "你叫什么名字"}]
    print(model(messages))
    
    
    
