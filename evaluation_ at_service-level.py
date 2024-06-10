import pandas as pd
import json

def df_trans(fault_data):
    ground_truth_records = []
    for hour_faults in fault_data.values():
        for fault in hour_faults:
            ground_truth_records.append({
                'inject_time': fault['inject_time'],
                'inject_pod': fault['inject_pod']
            })
    ground_truth_df = pd.DataFrame(ground_truth_records)
    return ground_truth_df

def evaluation(experiment_result, ground_truth_df):
    def calculate_pr_at_k(experiment_results, ground_truth_df, k):
        match_counts = 0

        # 遍历每个独特的注入时间
        for inject_time in ground_truth_df['inject_time'].unique():
            # 获取该时间的ground truth服务名列表
            gt_services = ground_truth_df[ground_truth_df['inject_time'] == inject_time]['inject_pod'].tolist()
            exp_results_at_time = experiment_results[experiment_results['InjectionTime'] == inject_time]

            # 如果实验结果中的top k个任何一个匹配上了ground truth中的任何一个服务
            if not exp_results_at_time.empty:
                top_k_results = exp_results_at_time.head(k)['ServiceName'].tolist()
                if any(result in gt_services for result in top_k_results):
                    match_counts += 1

        return match_counts

    # 计算PR@1, PR@3, PR@5
    pr_at_1_counts = calculate_pr_at_k(experiment_result, ground_truth_df, 1)
    pr_at_3_counts = calculate_pr_at_k(experiment_result, ground_truth_df, 3)
    pr_at_5_counts = calculate_pr_at_k(experiment_result, ground_truth_df, 5)
    return pr_at_1_counts, pr_at_3_counts, pr_at_5_counts

# 读取实验结果数据
experiment_results1_tt = pd.read_csv('raw_data/abnormal/suffer_anomaly_inject_data_1/result-8-22.csv')
experiment_results2_tt = pd.read_csv('raw_data/abnormal/suffer_anomaly_inject_data_2/result-8-23.csv')
experiment_results1_ob = pd.read_csv('raw_data/abnormal/suffer_anomaly_inject_data_3/result-1-29.csv')
experiment_results2_ob = pd.read_csv('raw_data/abnormal/suffer_anomaly_inject_data_4/result-1-30.csv')
# 读取ground truth数据
with open('raw_data/abnormal/suffer_anomaly_inject_data_1/2022-08-22-fault_list.json', 'r') as file1_tt:
    fault_data1_tt = json.load(file1_tt)
with open('raw_data/abnormal/suffer_anomaly_inject_data_2/2022-08-23-fault_list.json', 'r') as file2_tt:
    fault_data2_tt = json.load(file2_tt)
ground_truth_df1_tt = df_trans(fault_data1_tt)
ground_truth_df2_tt = df_trans(fault_data2_tt)
with open('raw_data/abnormal/suffer_anomaly_inject_data_3/2023-01-29-fault_list.json', 'r') as file1_ob:
    fault_data1_ob = json.load(file1_ob)
with open('raw_data/abnormal/suffer_anomaly_inject_data_4/2023-01-30-fault_list.json', 'r') as file2_ob:
    fault_data2_ob = json.load(file2_ob)
ground_truth_df1_ob = df_trans(fault_data1_ob)
ground_truth_df2_ob = df_trans(fault_data2_ob)
def calculate_pr(experiment_results1, experiment_results2, ground_truth_df1, ground_truth_df2):
    pr_at_1_counts = evaluation(experiment_results1, ground_truth_df1)[0] + evaluation(experiment_results2, ground_truth_df2)[0]
    pr_at_3_counts = evaluation(experiment_results1, ground_truth_df1)[1] + evaluation(experiment_results2, ground_truth_df2)[1]
    pr_at_5_counts = evaluation(experiment_results1, ground_truth_df1)[2] + evaluation(experiment_results2, ground_truth_df2)[2]
    lenth = len(ground_truth_df1['inject_time'].unique()) + len(ground_truth_df2['inject_time'].unique())

    pr_at_1 = pr_at_1_counts / lenth
    pr_at_3 = pr_at_3_counts / lenth
    pr_at_5 = pr_at_5_counts / lenth
    return pr_at_1, pr_at_3, pr_at_5

pr_at_1_ob, pr_at_3_ob, pr_at_5_ob = calculate_pr(experiment_results1_tt, experiment_results2_tt, ground_truth_df1_tt, ground_truth_df2_tt)
pr_at_1_tt, pr_at_3_tt, pr_at_5_tt =  calculate_pr(experiment_results1_ob, experiment_results2_ob, ground_truth_df1_ob,ground_truth_df2_ob)

print('---------------------------------------------')
print("Precision on OB:")
print('---------------------------------------------')
print(f'PR@1: {pr_at_1_ob: .2%} ')
print(f'PR@3: {pr_at_3_ob: .2%} ')
print(f'PR@5: {pr_at_5_ob: .2%} ')
print('---------------------------------------------')

print("Precision on TT:")
print('---------------------------------------------')
print(f'PR@1: {pr_at_1_tt: .2%} ')
print(f'PR@3: {pr_at_3_tt: .2%} ')
print(f'PR@5: {pr_at_5_tt: .2%} ')
print('---------------------------------------------')

