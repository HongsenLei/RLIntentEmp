# import json
# rm_cot_train_data_path="/seu_share/home/wutianxing/220222120/data/rm_sft/preference_data_CoT_62%_train.jsonl"

# def read_jsonl2list(file: str) -> list:
#     data = []
#     with open(file, 'r', encoding='utf-8') as read_file:
#         for line in read_file:
#             data.append(json.loads(line))
#     return data

# data = read_jsonl2list(rm_cot_train_data_path)
# num_yes = 0
# num_no = 0
# for item in data:
#     if item['label']=="Yes":
#         num_yes+=1
#     else:
#         num_no +=1
# print(num_yes,num_no)

a = [1.2,6356,48,45.6]
a = [x / 6 for x in a]
print(a)