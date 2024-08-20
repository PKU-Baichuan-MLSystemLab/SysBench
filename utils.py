import argparse
import json
import pandas as pd 
from collections import defaultdict
import re 
import numpy as np 

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# 定义计算加权平均的自定义函数
def weighted_mean(series):
    count = series.notna().sum()  # 非缺失值计数
    return series.sum() / count if count != 0 else np.nan

def analysis_eval_results(eval_result_filepath, analysis_eval_output_path):
    datas = json.load(open(eval_result_filepath, encoding="utf-8"))

    count_every_round = {i + 1: 0 for i in range(5)}
    count_continuous_round = {i + 1: 0 for i in range(5)} 
    count_continuous_round_relate = {i + 1: 0 for i in range(5)}  
    count_continuous_round_parallel = {i + 1: 0 for i in range(5)}  

    relate_align = {"align": [], "misalign": []}
    parallel_align = {"align": [], "misalign": []}
    total_align =  {"align": [], "misalign": []}

    count_relate = 0 
    count_parallel = 0 

    categorys_all = defaultdict(int)
    categorys_valid = defaultdict(int)

    all_infos = list() 
    datas = sorted(datas, key=lambda key:key["system_id"])

    for data in datas:
        if data["rounds_related"]:
            count_relate += 1
        else:
            count_parallel += 1
        session_flag = True 
        for index, message in enumerate([m for m in data["messages"] if m["role"] == "user"]):
            prompt = message["content"]
            eval_res = data["eval_results"][message["content"]]

            round_info = dict() 
            round_info["system_id"] = data["system_id"]
            round_info["领域"] = data["领域"]
            round_info["场景"] = data["场景"]
            round_info["multi_rounds_related"] = data["rounds_related"]
            round_info["alignment"] = data["prompt_infos"][prompt]["alignment"]
            round_info["round_index"] = index + 1
            round_info["system_prompt"] = data["system_prompt"]
            round_info["prompt"] = prompt
            round_info["referrence"] = data["messages"][index + 1]["content"]
            round_info["answer"] = eval_res["response"]
            round_info["infer_model"] = data["infer_model"]
            round_info["评判细则"] = "\n".join([str(cri["criteria_id"]) + ". " + cri["criteria_content"] + " | " + cri["criteria_type"]  for cri_index, cri in eval_res["criteria"].items()])
            round_info["评判理由"] = eval_res["评判理由"]
            round_info["评判结果"] = "\n".join([k + ". " + eval_res["criteria"][k]["criteria_type"] + " | " + v for k,v in eval_res["评判结果"].items()])
            round_info["是否可用"] = 1 if "否" not in eval_res["评判结果"].values() else 0

            # 统计多轮相关/平行-prompt对齐/冲突的可用率结果
            if round_info["multi_rounds_related"]:
                relate_align[round_info["alignment"]].append(round_info["是否可用"])
            else:
                parallel_align[round_info["alignment"]].append(round_info["是否可用"])
            total_align[round_info["alignment"]].append(round_info["是否可用"])
            
            # 统计多轮相关/平行-prompt对齐/冲突下的多轮连续遵循情况
            round_flag = True 
            for cri in eval_res["评判结果"]:
                categorys_all[eval_res["criteria"][cri]["criteria_type"]] += 1
                if eval_res["评判结果"][cri] == "是":
                    categorys_valid[eval_res["criteria"][cri]["criteria_type"]] += 1
                else:
                    round_flag = False 
            if round_flag:
                count_every_round[index + 1] += 1
                if session_flag:
                    count_continuous_round[index + 1] += 1
                    if round_info["multi_rounds_related"]:
                        count_continuous_round_relate[index + 1] += 1
                    else:
                        count_continuous_round_parallel[index + 1] += 1
            else:
                session_flag = False 
            
            all_infos.append(round_info)
    
    # 结果总表
    all_infos = pd.DataFrame(all_infos)
    # 每一轮的遵循率
    count_every_round_rate = {k: round(v/len(datas) * 100, 2) for k, v in count_every_round.items()}
    count_every_round["total"] = sum([v for _, v in count_every_round.items()])
    count_every_round_rate["total"] = round(count_every_round["total"] / (len(datas) * 5) * 100, 2)

    # 多轮连续遵循率
    count_continuous_round_relate_rate = {k: round(v/count_relate * 100, 2) for k, v in count_continuous_round_relate.items()}
    count_continuous_round_parallel_rate = {k: round(v/count_parallel * 100, 2) for k, v in count_continuous_round_parallel.items()}
    count_continuous_round_rate = {k: round(v/len(datas) * 100, 2) for k, v in count_continuous_round.items()}
    # 不同约束类型的可用率
    category_valid_rate = {type_key: round(categorys_valid[type_key] / categorys_all[type_key] * 100, 2) for type_key in categorys_all}
    categorys_all["total"] = sum(categorys_all.values())
    categorys_valid["total"] = sum(categorys_valid.values())
    category_valid_rate["total"] = round(categorys_valid["total"] / categorys_all["total"] * 100, 2)

    relate_align_total = {k: len(v) for k, v in relate_align.items()}
    parallel_align_total = {k: len(v) for k, v in parallel_align.items()}
    total_align_total = {k: len(v) for k, v in total_align.items()}
    
    relate_align_valid = {k: sum(v) for k, v in relate_align.items()}
    parallel_align_valid = {k: sum(v) for k, v in parallel_align.items()}
    total_align_valid = {k: sum(v) for k, v in total_align.items()}

    for align_item in [relate_align_total, parallel_align_total, total_align_total, relate_align_valid, parallel_align_valid, total_align_valid]:
        align_item["total"] = sum(align_item.values())
    
    relate_align_rate = {k: round(v/relate_align_total[k] * 100, 2) for k, v in relate_align_valid.items()}
    parallel_align_rate = {k: round(v/parallel_align_total[k] * 100, 2) for k, v in parallel_align_valid.items()}
    total_align_rate = {k: round(v/total_align_total[k] * 100, 2) for k, v in total_align_valid.items()}

    relate_align_rate["total"] = round(relate_align_valid["total"] / relate_align_total["total"] * 100, 2)
    parallel_align_rate["total"] = round(parallel_align_valid["total"] / parallel_align_total["total"] * 100, 2)
    total_align_rate["total"] = round(total_align_valid["total"] / total_align_total["total"] * 100, 2)

    print("=" * 50)
    print(f"多轮相关：{count_relate}, 多轮平行：{count_parallel}")
    print(json.dumps(count_continuous_round_relate, ensure_ascii=False, indent=2))
    print(json.dumps(count_continuous_round_parallel, ensure_ascii=False, indent=2))
    print(json.dumps(count_continuous_round, ensure_ascii=False, indent=2))
    print("-" * 50)
    print(json.dumps(count_continuous_round_relate_rate, ensure_ascii=False, indent=2))
    print(json.dumps(count_continuous_round_parallel_rate, ensure_ascii=False, indent=2))
    print(json.dumps(count_continuous_round_rate, ensure_ascii=False, indent=2))
    print("=" * 50)
    print(json.dumps(categorys_all, ensure_ascii=False, indent=2))
    print(json.dumps(categorys_valid, ensure_ascii=False, indent=2))
    print(json.dumps(category_valid_rate, ensure_ascii=False, indent=2))
    print("=" * 50)

    with pd.ExcelWriter(analysis_eval_output_path) as writer:
        # sheet1：详情
        all_infos.to_excel(writer, sheet_name='详情', index=False)
        # sheet2：不同约束类型遵循
        round_evals = pd.DataFrame([categorys_all, categorys_valid, category_valid_rate], index=["约束总量", "遵循数量", "遵循率"])
        cols = list(round_evals.columns)
        cols.remove("total")
        sorted(cols)
        cols.append("total")
        round_evals = round_evals[cols]
        round_evals.to_excel(writer, sheet_name='不同约束类型遵循')
        # sheet3：不同轮次遵循
        round_index = pd.DataFrame([count_every_round, count_every_round_rate], index=["当前轮次遵循数量", "遵循率"])
        round_index.to_excel(writer, sheet_name='不同轮次遵循')
        # sheet4：最大连续遵循轮次
        # count_continuous_round_relate["total"] = count_relate
        # count_continuous_round_parallel["total"] = count_parallel
        # count_continuous_round["total"] = count_relate + count_parallel
        
        # count_continuous_round_relate_rate["total"] = round(count_continuous_round_relate["total"] / count_relate * 100, 2)
        # count_continuous_round_parallel_rate["total"] = round(count_continuous_round_parallel["total"] / count_parallel * 100, 2)
        # count_continuous_round_rate["total"] = round(count_continuous_round["total"] / (count_relate + count_parallel) * 100, 2)
        def cal_continuous_avg(item, total):
            continuous_count = list(item.values())
            continuous_count.append(0)
            continuous_count.insert(0, total)
            return round(sum([(continuous_count[i] - continuous_count[i + 1]) * i for i in range(len(continuous_count) - 1)]) / total, 2)
            
        count_continuous_round_relate_rate["total"] = cal_continuous_avg(count_continuous_round_relate, count_relate)
        count_continuous_round_parallel_rate["total"] = cal_continuous_avg(count_continuous_round_parallel, count_parallel)
        count_continuous_round_rate["total"] = cal_continuous_avg(count_continuous_round, count_relate + count_parallel)
        
        for item in [count_continuous_round_relate, count_continuous_round_parallel, count_continuous_round]:
            item["total"] = round(sum(k * v for k, v in item.items()) / len(item), 2)

        merge_continuous_round = pd.concat([pd.DataFrame([count_continuous_round_relate, count_continuous_round_relate_rate]), 
                                            pd.DataFrame([count_continuous_round_parallel, count_continuous_round_parallel_rate]), 
                                            pd.DataFrame([count_continuous_round, count_continuous_round_rate])], axis=1)
        merge_continuous_round.to_excel(writer, sheet_name='最大连续遵循轮次')

        # sheet5：统计多轮相关/平行-prompt对齐/冲突的可用率结果
        merge_sys_prompt_align = pd.concat([pd.DataFrame([relate_align_total, relate_align_valid, relate_align_rate]), 
                                            pd.DataFrame([parallel_align_total, parallel_align_valid, parallel_align_rate]), 
                                            pd.DataFrame([total_align_total, total_align_valid, total_align_rate])], axis=1)
        merge_sys_prompt_align.to_excel(writer, sheet_name='统计多轮相关|平行-prompt对齐|冲突的可用率结果')


def get_eval_pattern(messages, criteria):
    system_prompt = messages[0]["content"]
    historys = messages[1:-2]
    current_dialogs = messages[-2::]
    criteria_pattern = "\n".join([str(criteria[cri_id]["criteria_id"]) + ". " + criteria[cri_id]["criteria_content"] + " | " + criteria[cri_id]["criteria_type"] for cri_id in criteria])

    pattern = f"""
# 背景与目标
你现在是一个大模型结果评估专家。以下，你将会面对一个大模型system prompt遵循能力评估任务。
我将会给出你对应的system prompt、历史对话、当前轮次的问题、和当前轮次的回答
需要你准确判断当前轮次的回答是否合格。
为了让你的评判足够准确，我还将会给出评判细则，需要你准确判断并告知评判细则当中每一条约束的遵循情况。

# 对话信息
## system prompt
<system prompt>
{system_prompt}
</system prompt>

## 历史对话轮次

"""
    
    for index in range(len(historys) // 2):
        user_message = historys[index * 2]
        assistant_message = historys[index * 2 + 1]

        pattern += f"""
<round-{index + 1}>
<role:>
{user_message["role"]}
</role>
<content>
{user_message["content"]}
</content>
<role:>{assistant_message["role"]}</role>
<content>
{assistant_message["content"]}
</content>
</round-{index + 1}>

""" 

    pattern += f"""
## 当前待评估的对话轮次
<role:>
{current_dialogs[0]["role"]}
</role>
<content>
{current_dialogs[0]["content"]}
</content>

<role:>
{current_dialogs[1]["role"]}
</role>
<content>
{current_dialogs[1]["content"]}
</content>

# 评判细则
<评判细则>
{criteria_pattern}
</评判细则>
"""
    
    pattern += """
请你认真阅读上述system prompt设定与历史对话轮次，并严格以评判细则为评判标准，针对评判细则当中的逐条要求，判断当前对话轮次的回答是否遵循。
请以json格式回答，包含两个字段：评判理由、评判结果（评判结果为一个dict，dict的key评判细则的序号，value为对应的评判结果（是/否））
输出格式如下：
'''json
{
  "评判理由": "……",
  "评判结果": {
    1: "……",
    ……
  }
}
'''

"""
    return pattern 


def character_count(answer, criteria_content):
    chinese_character_count = re.findall(r"汉字字数数量(|大于|小于|等于)(\d{1,5})", criteria_content)
    character_count = re.findall(r"字数数量(|大于|小于|等于)(\d{1,5})", criteria_content)
    assert len(chinese_character_count) <= 0
    assert len(character_count) <= 0 

    if len(chinese_character_count) > 0:
        comparison, number = chinese_character_count[0]
        number = int(number)
        chinese_character = re.findall("[\u4e00-\u9fa5]", answer)
        if comparison == "大于":
            return len(chinese_character) > number
        elif comparison == "等于":
            return len(chinese_character) == number
        else:
            return len(chinese_character) < number
    elif len(character_count) > 0:
        comparison, number = character_count[0]
        number = int(number)
        if comparison == "大于":
            return len(answer) > number
        elif comparison == "等于":
            return len(answer) == number
        else:
            return len(answer) < number
    else:
        return -1 
