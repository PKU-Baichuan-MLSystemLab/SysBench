import sys
import json 
import random 
from zhipuai import ZhipuAI

class glm4(object):
    def __init__(self, model="glm-4-0520") -> None:
        self.model = model
    
    def __call__(self, messages):
        client = ZhipuAI(api_key="<enter your key>")
        response = client.chat.completions.create(
        model=self.model,  # 填写需要调用的模型编码
            messages=messages,
            stream=False,
            )
        return response.choices[0].message.content
        
if __name__ == "__main__":
    model = glm4()    
    messages = [{"role": "system", "content": "你是百灵鸟，你是一个给人看病的医生"}, {"role": "user", "content": "你叫什么名字"}]
    print(model(messages))
    
    
    
