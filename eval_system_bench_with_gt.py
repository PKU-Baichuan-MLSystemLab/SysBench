import json
import argparse
import concurrent.futures
import os 
import random 
import traceback
import sys 
import threading 
import importlib
from  models import * 
from utils import *

class SystemBenchEval():
    def __init__(self, infer_model_name, infer_with_gt_history, eval_model_name, eval_dataset_path, output_dir):
        self.infer_with_gt_history = infer_with_gt_history
        if self.infer_with_gt_history:
            self.output_dir = os.path.join(output_dir, f"{infer_model_name}_with_gt_history")
        else:
            self.output_dir = os.path.join(output_dir, infer_model_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.infer_model_name = infer_model_name
        self.infer_model = self.get_model_class(infer_model_name)
        self.eval_model_name = eval_model_name
        self.eval_model = self.get_model_class(eval_model_name)
        self.eval_dataset_path = eval_dataset_path

        self.infer_output_path = os.path.join(self.output_dir, f"{infer_model_name}_infer.json")
        self.eval_output_path = os.path.join(self.output_dir, f"{infer_model_name}_eval.json")
        self.analysis_result_output_path = os.path.join(self.output_dir, f"{infer_model_name}_analysis.xlsx") 

    def get_model_class(self, model_type):
        module_name = f"models.{model_type}"
        class_name = f"{model_type}"
        print(module_name, class_name)
        try:
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            return model_class()
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Model type '{model_type}' is not defined: {e}")
    
    def load_examples(self, dataset_filepath):
        try:
            datas = json.load(open(dataset_filepath, encoding="utf-8"))
        except:
            return list()
        return datas 

    def do_infer(self, data, retry_time=10):
        all_messages = list()
        for index, mess in enumerate(data["messages"]):
            if mess["role"] in {"system", "user"}:
                pass 
            else:
                messages = data["messages"][0:index]
                retry_i = 0 
                while retry_i < retry_time:
                    try:
                        response = self.infer_model(messages)
                        assert response is not None and isinstance(response, str)
                        messages.append({"role": "assistant", "content": response})
                        all_messages.append(messages)
                        break
                    except Exception as e:
                        traceback.print_exc()
                    retry_i += 1
                else:
                    raise 
        
        assert len(all_messages) == len(data["messages"]) // 2
        data["infer_model"] = self.infer_model_name
        data["infer_results"] = all_messages

        return data 

    def do_eval(self, data, retry_time=10):
        eval_results = dict() 
        for messages in data["infer_results"]:
            prompt = messages[-2]["content"]
            answer = messages[-1]["content"]
            criteria = data["prompt_infos"][messages[-2]["content"]]["criteria"]
            eval_pattern = get_eval_pattern(messages=messages, criteria=criteria)

            retry_i = 0 
            while retry_i < retry_time:
                try:
                    eval_response = self.eval_model([{"role": "user", "content": eval_pattern}], temperature=0).strip()
                    eval_response_js = eval(eval_response[7:-3])
                    assert "评判理由" in eval_response_js and "评判结果" in eval_response_js and isinstance(eval_response_js["评判结果"], dict), eval_response

                    assert set([int(n) for n in eval_response_js["评判结果"].keys()]) == set([int(ck) for ck in criteria]), "-" * 50  + eval_pattern + "\n" + "-" * 50 + json.dumps(eval_response_js, ensure_ascii=False, indent=2) + "\n" + "-" * 50 + json.dumps(criteria, ensure_ascii=False, indent=2) + "-" * 50
                    all(value in {"是", "否"} for value in eval_response_js["评判结果"].values())
                    eval_response_js["eval_pattern"] = eval_pattern
                    eval_response_js["response"] = answer
                    eval_response_js["criteria"] = criteria
                    eval_response_js["retry_time"] = retry_i

                    for cr in criteria:
                        check_res = character_count(criteria[cr]["criteria_content"], answer)
                        if check_res == -1:
                            continue 
                        else:
                            if check_res is True:
                                eval_response_js["评判结果"][cr] = "是"
                            else:
                                eval_response_js["评判结果"][cr] = "否"

                    eval_results[messages[-2]["content"]] = eval_response_js 
                    break 
                except Exception as e:
                    print(eval_response)
                    print(json.dumps(criteria, ensure_ascii=False, indent=2))
                    print(eval_pattern)
                    traceback.print_exc()
                retry_i += 1
            else:
                data["eval_results"] = None 
                return data 
        
        data["eval_results"] = eval_results
        return data 
    
    def execute(self, do_infer=False, do_eval=False, max_threads=20):
        def worker(task_type, input_path, output_path):
            datas = self.load_examples(input_path)
            total_count = len(datas)
            
            cache_filepath = output_path + "_cache.json"
            if os.path.exists(cache_filepath):
                cache_datas = [data for data in self.load_examples(cache_filepath) if data[f"{task_type}_results"] is not None]
            else:
                cache_datas = list() 
            completed = len(cache_datas)

            print(f"{task_type}: {completed} / {total_count}")
            cache_data_ids = [data["system_id"] for data in cache_datas]
            rest_datas = [data for data in datas if data["system_id"] not in cache_data_ids]
            random.shuffle(rest_datas)

            lock = threading.Lock()
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                tasks = {executor.submit(getattr(self, f"do_{task_type}"), rest_data) for rest_data in rest_datas}
                for future in concurrent.futures.as_completed(tasks):
                    result = future.result()
                    try:
                        result = future.result()
                    except Exception as e:
                        traceback.print_exc()
                    else:
                        with lock:
                            cache_datas.append(result)
                            json.dump(cache_datas, open(cache_filepath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                            if result[f"{task_type}_results"] is not None:
                                completed += 1
                        
                        print(f"{task_type}: {completed} / {total_count}")
            
            json.dump(cache_datas, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            assert total_count - completed == 0, f"失败数量：{total_count - completed}"
        
        if do_infer:
            print("do infer")
            worker("infer", self.eval_dataset_path, self.infer_output_path)
        print("-" * 50)
        if do_eval:
            print("do eval")
            worker("eval", self.infer_output_path, self.eval_output_path)
            analysis_eval_results(eval_result_filepath=self.eval_output_path, analysis_eval_output_path=self.analysis_result_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_model_name", type=str)
    parser.add_argument("--infer_with_gt_history", type=str2bool, default=False, help="true or false")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--max_threads", type=int, default=100)
    args = parser.parse_args()

    eval_model_name = "gpt4o"
    eval_dataset_path = "datas/system_benchmark_eval_datas.json"    
    output_dir = args.output_dir

    system_bench_eval = SystemBenchEval(infer_model_name=args.infer_model_name, infer_with_gt_history=args.infer_with_gt_history, eval_model_name=eval_model_name, eval_dataset_path=eval_dataset_path, output_dir=output_dir)
    system_bench_eval.execute(do_infer=True, do_eval=True, max_threads=args.max_threads)
