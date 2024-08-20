from modelscope import AutoModelForCausalLM, AutoTokenizer
device = "cuda" 

class qwen2_72b():
    def __init__(self, model_path="/path/to/Qwen2-72B-Instruct"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def __call__(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=2048
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


if __name__ == "__main__":
    messages = [{"role": "system", "content": "你是百灵鸟，你是一个给人看病的医生"}, {"role": "user", "content": "你叫什么名字"}]

    qwen2_72b_model = qwen2_72b()
    print(qwen2_72b_model(messages))

