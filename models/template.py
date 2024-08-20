
class template_model():
    def __init__(self):
        pass 

    def __call__(self, messages):
        raise NotImplementedError("please implement the __call__ method")

if __name__ == "__main__":
    messages = [{"role": "system", "content": "You are a doctor named Jack"}, {"role": "user", "content": "What's your name"}]
    model = template_model()
    print(model(messages))
    