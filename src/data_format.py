import json
import copy

prefix = "<image>\n"
def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析每一行的JSON对象并添加到列表中
            data.append(json.loads(line))
    return data


def count_image_newline(data):
    """
    递归遍历JSON数据，统计所有包含"image\n"的字符串的出现次数。

    参数:
    data (dict or list or str): 输入的JSON数据，可以是字典、列表或字符串。

    返回:
    int: "image\n"字符串的总数量。
    """
    count = 0

    if isinstance(data, dict):
        for value in data.values():
            count += count_image_newline(value)
    elif isinstance(data, list):
        for item in data:
            count += count_image_newline(item)
    elif isinstance(data, str):
        count += data.count(prefix)
    
    return count

def concat(file_list):
    data = []
    for file in file_list:
        if file.endswith("jsonl"):
            data = data + read_jsonl_file(file)
        else:
            data = data + json.load(open(file,'r'))
        
    return data

def format_data(raw_data):
    #for index in range(len(raw_data)):
    processed_data = []
    for item in raw_data:
        #先验证图片部分
        if isinstance(item["image"],list):
            assert count_image_newline(item) == len(item["image"]), "多图数据中，图像数量 和 '<image>\n'数量应保持一致！"   
        else:
            assert count_image_newline(item) == 1, "单图数据中，'<image>\n'数量应为1！"  

        for i in range(len(item["conversations"])):
            #把角色的名称改了
            try:
                if item["conversations"][i]["from"] == "user":
                    item["conversations"][i]["from"] = "human"
                    #del item["conversations"][i]["from"]
                elif item["conversations"][i]["from"] == "assistant":
                    item["conversations"][i]["from"] = "gpt"
                    #del item["conversations"][i]["from"]
            except:
                print(item)
        #结尾不能是human
        if item["conversations"][-1]["from"] == "human":
            item["conversations"].pop()
            #print(bad_info["from"])
        if len(item["conversations"])%2 !=0:
            continue
        processed_data.append(item)

    return processed_data

def write_data(output_file,data):
    with open(output_file,'w') as f:
        f.write(json.dumps(data,indent=2,ensure_ascii=False))
        

if __name__ == "__main__":

    #json_list=["/work/home/acehekbmzh/cxx/temp/train_data_1130_part.jsonl"]
    json_list=["/mnt/afs/chenxiaoxuan/modality_alignment/Qwen2-VL-Finetune/data/train_data_1130_part.jsonl"]
    #output_file = "/work/home/acehekbmzh/cxx/temp/train_data_1130_part_format.json"
    output_file = "/mnt/afs/chenxiaoxuan/modality_alignment/Qwen2-VL-Finetune/data/train_data_1130_part_format.json"

    all_data_raw = concat(json_list)
    data_processed = format_data(all_data_raw)
    print(f"All data: {len(data_processed)}")
    #print(data_processed[105])
    write_data(output_file,data_processed)
