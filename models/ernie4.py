import sys
import json 
import random 
import requests
import json

random.seed(1234)


class ernie4(object):
    def __init__(self) -> None:
        pass 

    def __call__(self, messages, system=None, temperature=0.95, finish_try=2, keys=None):
        if isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]
        if messages[0]["role"] in {"system", "user_system"}:
            system = messages[0]["content"]
            messages = messages[1:]
        
        assert isinstance(messages, list)
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-4.0-8k-0613?access_token=" + self.get_access_token(keys)
        payload = {
            "messages": messages,
            "top_p":0.8,
            "penalty_score": 1,
            "temperature": 0.95,
            "disable_search": False,
            "enable_citation": False}
        if system:
            payload["system"] = system
        payload = json.dumps(payload)
        headers = {'Content-Type': 'application/json'}
        while True:
            try:
                response = requests.request("POST", url, headers=headers, data=payload)
                assert response.status_code == 200
                response = json.loads(response.text)
                print(f"response:{response}")

                response = response["result"]
                finish_try -= 1

                return response
            except Exception as e:
                print(f"【response】:{response}\t【Error】:{e}", flush=True)
                try:
                    error_code = json.loads(e.http_body)['error']['code']
                    if error_code in ('billing_not_active', 'context_length_exceeded'):
                        return '', error_code
                except:
                    pass
                return None
        

    def get_access_token(self, keys=None):
        """
        使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
        """ 
        API_SECRET_KEYs = [["client_id", "client_secret"]]
        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={API_SECRET_KEYs[0][0]}&client_secret={API_SECRET_KEYs[0][1]}"
        payload = json.dumps("")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json().get("access_token")


if __name__ == "__main__":
    model = ernie4()
    print(model([{"role": "system", "content": "你是百灵鸟，你是一个给人看病的医生"}, {"role": "user", "content": "你叫什么名字"}]))
