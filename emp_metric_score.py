from tqdm import tqdm
from typing import Dict, Optional, Sequence, List
from src.constant import ANALYSE_PATTERN,ANSWER_TRIGGER
from src.utils import write_jsonl_append_line,read_jsonl2list
import re
import argparse
import os
from bert_score import BERTScorer
from typing import List, Tuple
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

# 定义要提取的指标及其对应的折线分组
conv_metrics_groups = {
    'intent_rouge_l': [
        'intent_rouge_l_precision',
        'intent_rouge_l_recall',
        'intent_rouge_l_f1'
    ],
    'intent_levenshtein_distance': [
        'intent_levenshtein_distance'
    ],
    'response_rouge_l': [
        'response_rouge_l_precision',
        'response_rouge_l_recall',
        'response_rouge_l_f1'
    ],
    'response_levenshtein_distance': [
        'response_levenshtein_distance'
    ],
    'response_distinct': [
        'response_distinct_1',
        'response_distinct_2'
    ],
    'response_bleu': [
        'response_bleu_2',
        'response_bleu_3',
        'response_bleu_4'
    ],
    'response_bert_score':[
        'response_bert_score_R',
        'response_bert_score_P',
        'response_bert_score_F1'
    ]
}

rm_metrics_groups={
    'correct': [
        'correct'
    ],
    'reward': [
        'reward'
    ]
}

def lcs(reference, prediction):
    len_ref, len_pred = len(reference), len(prediction)
    if len_ref == 0 or len_pred == 0:
        return 0
    # 创建二维数组，存储LCS长度
    dp = [[0] * (len_pred + 1) for _ in range(len_ref + 1)]
    
    # 动态规划计算LCS
    for i in range(1, len_ref + 1):
        for j in range(1, len_pred + 1):
            if reference[i - 1] == prediction[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[len_ref][len_pred]

def rouge_l(reference, prediction):
    if type(reference) is str and type(prediction) is str:
        reference = reference.split()
        prediction = prediction.split()
    len_ref = len(reference)
    len_pred = len(prediction)
    
    # 边界条件：如果其中一个序列为空
    if len_ref == 0 and len_pred == 0:
        return 1.0, 1.0, 1.0
    if len_ref == 0 or len_pred == 0:
        return 0.0, 0.0, 0.0
    
    # 计算LCS长度
    lcs_length = lcs(reference, prediction)
    
    # 计算精确率、召回率和F1分数
    recall = lcs_length / len_ref if len_ref > 0 else 0
    precision = lcs_length / len_pred if len_pred > 0 else 0
    if recall + precision == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1


def levenshtein_distance(reference, prediction):
    if type(reference) is str and type(prediction) is str:
        reference = reference.split()
        prediction = prediction.split()
    len_ref, len_pred = len(reference), len(prediction)
    
    # 边界条件：如果其中一个序列为空
    if len_ref == 0:
        return len_pred  # 需要插入所有的预测元素
    if len_pred == 0:
        return len_ref  # 需要删除所有的参考元素
    
    # 创建二维数组，存储编辑距离
    dp = [[0] * (len_pred + 1) for _ in range(len_ref + 1)]
    
    # 初始化边界条件
    for i in range(len_ref + 1):
        dp[i][0] = i
    for j in range(len_pred + 1):
        dp[0][j] = j
    
    # 动态规划计算编辑距离
    for i in range(1, len_ref + 1):
        for j in range(1, len_pred + 1):
            if reference[i - 1] == prediction[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 无需操作
            else:
                dp[i][j] = min(dp[i - 1][j],    # 删除
                               dp[i][j - 1],    # 插入
                               dp[i - 1][j - 1] # 替换
                              ) + 1
    
    return dp[len_ref][len_pred]



def bleu_n(reference: List[int], prediction: List[int]) -> Tuple[float, float, float]:
    """计算BLEU-2、BLEU-3、BLEU-4"""
    bleu_2 = sentence_bleu([reference],prediction,weights=[0,1,0,0])
    bleu_3 = sentence_bleu([reference],prediction,weights=[0,0,1,0])
    bleu_4 = sentence_bleu([reference],prediction,weights=[0,0,0,1])
    return bleu_2, bleu_3, bleu_4

def calc_distinct_n(n, ckpt_data):
    dict = {}
    total = 0
    candidates = [data_dict['predict_response_token_ids'] for data_dict in ckpt_data]
    for sentence in candidates:
        for i in range(len(sentence) - n + 1):
            ney = tuple(sentence[i : i + n])
            dict[ney] = 1
            total += 1
    score = len(dict) / (total + 1e-16)
    return score

def main(args):
    if args.data_mode == "conv":
        bert_scorer = BERTScorer(model_type=args.bert_path,num_layers=17, batch_size=1)
        metrics_groups = conv_metrics_groups
    elif args.data_mode == "rm":
        metrics_groups = rm_metrics_groups

    # 用于存储数据
    checkpoints = []
    
    # 遍历目录中的所有子目录，找到符合要求的文件
    for subdir in os.listdir(args.exp_model_path):
        subdir_path = os.path.join(args.exp_model_path, subdir)
        
        # 确保是目录且符合 checkpoint-XXX 的命名规则
        match = re.match(r'checkpoint-(\d+)', subdir)  # 提取数字部分
        if os.path.isdir(subdir_path) and match:
            # 提取数字并转换为整数
            checkpoint_number = int(match.group(1))
            checkpoints.append((checkpoint_number, subdir_path))  # 存储数字和目录名


            # 按照数字大小排序 checkpoints
    checkpoints.sort(key=lambda x: x[0])  # 根据 checkpoint_number 排序
    exp_metric = {}

    for checkpoint_number, subdir_path in checkpoints:
        # 对每个ckpt求评测指标
        eval_file = os.path.join(subdir_path, f'{args.data_mode}_{args.conv_sample_mode}_eval.jsonl')
        metric_score_file = os.path.join(subdir_path, f'{args.data_mode}_{args.conv_sample_mode}_metric_score.jsonl')
        if os.path.exists(metric_score_file):
            os.remove(metric_score_file)
            print(f"Old file '{metric_score_file}' has been removed!")
        sample_data = read_jsonl2list(eval_file) if os.path.exists(eval_file) else []
        ckpt_metric = {metric: [] for group in metrics_groups.values() for metric in group}
        if args.data_mode == "conv":
            for sam_da in sample_data:
                
                intent_rouge_l_precision, intent_rouge_l_recall, intent_rouge_l_f1 = rouge_l(sam_da['reference_intent_list'],sam_da['predicted_intent_list']) # precision, recall, f1
                intent_levenshtein_distance = levenshtein_distance(sam_da['reference_intent_list'],sam_da['predicted_intent_list'])
                response_rouge_l_precision, response_rouge_l_recall, response_rouge_l_f1 = rouge_l(sam_da['reference_response_token_ids'], sam_da['predict_response_token_ids'])
                response_levenshtein_distance = levenshtein_distance(sam_da['reference_response_token_ids'], sam_da['predict_response_token_ids'])
                response_bleu_2, response_bleu_3, response_bleu_4 = bleu_n(sam_da['reference_response_token_ids'], sam_da['predict_response_token_ids'])
                response_bert_score_P,response_bert_score_R,response_bert_score_F1 = bert_scorer.score([sam_da['predict_response']],[sam_da['reference_response']], batch_size=1)

                # 一次性
                sam_da_metric={
                    'intent_rouge_l_precision':intent_rouge_l_precision,
                    'intent_rouge_l_recall':intent_rouge_l_recall,
                    'intent_rouge_l_f1':intent_rouge_l_f1,
                    'intent_levenshtein_distance':intent_levenshtein_distance,
                    'response_rouge_l_precision':response_rouge_l_precision,
                    'response_rouge_l_recall':response_rouge_l_recall,
                    'response_rouge_l_f1':response_rouge_l_f1,
                    'response_levenshtein_distance':response_levenshtein_distance,
                    'response_bleu_2':response_bleu_2,
                    'response_bleu_3':response_bleu_3,
                    'response_bleu_4':response_bleu_4,
                    'response_bert_score_P':response_bert_score_P.item(),
                    'response_bert_score_R':response_bert_score_R.item(),
                    'response_bert_score_F1':response_bert_score_F1.item()
                }
                # print(sam_da_metric)
                sam_da['metric'] = sam_da_metric
                write_jsonl_append_line(metric_score_file,sam_da)

                for k,v in sam_da_metric.items():
                    ckpt_metric[k].append(v)
            # 全局的，每个ckpt
            ckpt_metric['response_distinct_1'] = [calc_distinct_n(1, sample_data)]
            ckpt_metric['response_distinct_2'] = [calc_distinct_n(2, sample_data)]
        elif args.data_mode == "rm": 
            for sam_da in sample_data:
                correct = 1 if sam_da['CoT_label'] == sam_da['predicted_CoT_label'] else 0
                
                #########
                # 近似奖励
                #########
                _response = sam_da['response']
                _sample = sam_da['predict_CoT']
                response_intent = re.findall(r"<\|([^|]+)\|>", _response)
                response_intent = response_intent[1:-1] # 去掉<|Intent begin|> <|Intent end|>
                cleaned_message = [part.strip() for part in re.split(r'<\|.*?\|>', _response) if part.strip()]
                assert len(response_intent)==len(cleaned_message)
                num_step = len(response_intent)
                reward = 0.0
                if correct: # 最终预测正确
                    reward += 0.5
                    matches = ANALYSE_PATTERN.findall(_sample)
                    if len(matches)==num_step:
                        reward += 0.1 # 步骤分析正确, 剩余分析奖励最高0.4
                        for stp, match in enumerate(matches):
                            m_step, m_intent, m_content, _ = match # TODO
                            if int(m_step)-1==stp:
                                reward += (0.4/num_step)*0.1
                            if m_intent.strip() == response_intent[stp].strip():
                                reward += (0.4/num_step)*0.6
                            if m_content.strip() == cleaned_message[stp].strip():
                                reward += (0.4/num_step)*0.3
                sam_da_metric={
                    'correct':correct,
                    'reward':reward
                }
                sam_da['metric'] = sam_da_metric
                write_jsonl_append_line(metric_score_file,sam_da)

                for k,v in sam_da_metric.items():
                    ckpt_metric[k].append(v)
       
        result_file = os.path.join(subdir_path, f"total_res_{args.data_mode}_{args.conv_sample_mode}_eval")
        with open(result_file, 'w') as f:
            # 遍历 total_metirc 字典并写入文件
            for k, v in ckpt_metric.items():
                average_value = sum(v) / len(v) if len(v) > 0 else 0
                # 将格式化的结果写入文件
                f.write(f"{k}: {average_value}\n")
                print(f"checkpoint-{checkpoint_number} {k}: {average_value}\n")
                ckpt_metric[k] = average_value
        exp_metric[checkpoint_number] = ckpt_metric
    
    # 画图
    print(exp_metric)
    paint_dir = os.path.join(args.exp_model_path, f"metric_paint_{args.conv_sample_mode}")
    os.makedirs(paint_dir, exist_ok=True) 
    for title, metrics in metrics_groups.items():
        plt.figure(figsize=(10, 6))  # 创建新的图形
        plt.title(title)
        for metric in metrics:
            paint_metric_values = [exp_metric[ckpt_num][metric] for ckpt_num in exp_metric.keys()]
            plt.plot(exp_metric.keys(), paint_metric_values, label=metric)
            # 在每个数据点上显示其 y 值
            for i, (ckpt_num, value) in enumerate(zip(exp_metric.keys(), paint_metric_values)):
                plt.text(ckpt_num, value, f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        plt.xlabel('Checkpoint Number')
        plt.ylabel('Metric Value')
        plt.legend()
        
        # 保存并显示图
        plt.savefig(os.path.join(paint_dir, f'{title}.png'))
        # plt.show()
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute all ckpt score and paint")
    parser.add_argument('--data_mode', type=str,required=True)
    parser.add_argument('--conv_sample_mode', type=str, default='valid') # conv_sample_mode: valid/test
    parser.add_argument('--exp_model_path', type=str, required=True)
    parser.add_argument('--bert_path', type=str, default="/root/autodl-tmp/model/roberta-large")
    # 解析命令行参数
    args = parser.parse_args()
    main(args)

# python emp_metric_score.py --data_mode "conv" --conv_sample_mode "test" --exp_model_path "/root/autodl-tmp/experients/ppo_origin_lr_5e-6_vm" 
# python emp_metric_score.py --data_mode "rm" --exp_model_path "/seu_share/home/wutianxing/220222120/experients/sft_rm_lr_5e-6_bz_128"