import pandas as pd
import json

# Function to transform fault data into a DataFrame
def df_trans(fault_data):
    ground_truth_records = []
    for hour_faults in fault_data.values():
        for fault in hour_faults:
            ground_truth_records.append({
                'inject_time': fault['inject_time'],
                'inject_pod': fault['inject_pod'],
                'inject_type': fault['inject_type']
            })
    ground_truth_df = pd.DataFrame(ground_truth_records)
    return ground_truth_df

# Function to evaluate PR at K with additional conditions
def evaluation(experiment_result, ground_truth_df):
    def calculate_pr_at_k(experiment_results, ground_truth_df, k):
        match_counts = 0

        # Iterate over each unique inject time
        for inject_time in ground_truth_df['inject_time'].unique():
            # Get the ground truth services and types for the inject time
            gt_data = ground_truth_df[ground_truth_df['inject_time'] == inject_time]
            gt_services = gt_data['inject_pod'].tolist()
            gt_types = gt_data['inject_type'].tolist()

            # Get the experiment results for the inject time, sorted by PValue descending
            exp_results_at_time = experiment_results[experiment_results['InjectionTime'] == inject_time]

            if not exp_results_at_time.empty:
                top_k_results = exp_results_at_time.head(k)

                # Check if any of the top K results match the ground truth services and types
                for index, row in top_k_results.iterrows():
                    if row['ServiceName'] in gt_services and row['anomaly_type'] in gt_types:
                        match_counts += 1
                        break

        return match_counts

    # Calculate PR@1, PR@3, PR@5
    pr_at_1_counts = calculate_pr_at_k(experiment_result, ground_truth_df, 1)
    pr_at_3_counts = calculate_pr_at_k(experiment_result, ground_truth_df, 3)
    pr_at_5_counts = calculate_pr_at_k(experiment_result, ground_truth_df, 5)
    return pr_at_1_counts, pr_at_3_counts, pr_at_5_counts

# Load experiment results
experiment_results1_tt = pd.read_csv('RCA/result-1-29.csv')
experiment_results2_tt = pd.read_csv('RCA/result-1-30.csv')

# Load ground truth data
with open('raw_data/abnormal/suffer_anomaly_inject_data_3/2023-01-29-fault_list.json', 'r') as file1:
    fault_data1_tt = json.load(file1)
with open('raw_data/abnormal/suffer_anomaly_inject_data_4/2023-01-30-fault_list.json', 'r') as file2:
    fault_data2_tt = json.load(file2)

# Transform ground truth data
ground_truth_df1_tt = df_trans(fault_data1_tt)
ground_truth_df2_tt = df_trans(fault_data2_tt)

# Exclude entries with inject_type 'return' and 'exception'
filtered_ground_truth_df1_tt = ground_truth_df1_tt[~ground_truth_df1_tt['inject_type'].isin(['return', 'exception'])]
filtered_ground_truth_df2_tt = ground_truth_df2_tt[~ground_truth_df2_tt['inject_type'].isin(['return', 'exception'])]

# Evaluate PR at K
pr_at_1_counts_tt = evaluation(experiment_results1_tt, filtered_ground_truth_df1_tt)[0] + evaluation(experiment_results2_tt, filtered_ground_truth_df2_tt)[0]
pr_at_3_counts_tt = evaluation(experiment_results1_tt, filtered_ground_truth_df1_tt)[1] + evaluation(experiment_results2_tt, filtered_ground_truth_df2_tt)[1]
pr_at_5_counts_tt = evaluation(experiment_results1_tt, filtered_ground_truth_df1_tt)[2] + evaluation(experiment_results2_tt, filtered_ground_truth_df2_tt)[2]
lenth_tt = len(filtered_ground_truth_df1_tt['inject_time'].unique()) + len(filtered_ground_truth_df2_tt['inject_time'].unique())

pr_at_1_tt = pr_at_1_counts_tt / lenth_tt
pr_at_3_tt = pr_at_3_counts_tt / lenth_tt
pr_at_5_tt = pr_at_5_counts_tt / lenth_tt

experiment_results1_ob = pd.read_csv('RCA/result-8-22.csv')
experiment_results2_ob = pd.read_csv('RCA/result-8-23.csv')

# Load ground truth data
with open('raw_data/abnormal/suffer_anomaly_inject_data_1/2022-08-22-fault_list.json', 'r') as file1:
    fault_data1_ob = json.load(file1)
with open('raw_data/abnormal/suffer_anomaly_inject_data_2/2022-08-23-fault_list.json', 'r') as file2:
    fault_data2_ob = json.load(file2)

# Transform ground truth data
ground_truth_df1_ob = df_trans(fault_data1_ob)
ground_truth_df2_ob = df_trans(fault_data2_ob)

# Exclude entries with inject_type 'return' and 'exception'
filtered_ground_truth_df1_ob = ground_truth_df1_ob[~ground_truth_df1_ob['inject_type'].isin(['return', 'exception'])]
filtered_ground_truth_df2_ob = ground_truth_df2_ob[~ground_truth_df2_ob['inject_type'].isin(['return', 'exception'])]

# Evaluate PR at K
pr_at_1_counts_ob = evaluation(experiment_results1_ob, filtered_ground_truth_df1_ob)[0] + evaluation(experiment_results2_ob, filtered_ground_truth_df2_ob)[0]
pr_at_3_counts_ob = evaluation(experiment_results1_ob, filtered_ground_truth_df1_ob)[1] + evaluation(experiment_results2_ob, filtered_ground_truth_df2_ob)[1]
pr_at_5_counts_ob = evaluation(experiment_results1_ob, filtered_ground_truth_df1_ob)[2] + evaluation(experiment_results2_ob, filtered_ground_truth_df2_ob)[2]
lenth_ob = len(filtered_ground_truth_df1_ob['inject_time'].unique()) + len(filtered_ground_truth_df2_ob['inject_time'].unique())

pr_at_1_ob = pr_at_1_counts_ob / lenth_ob
pr_at_3_ob = pr_at_3_counts_ob / lenth_ob
pr_at_5_ob = pr_at_5_counts_ob / lenth_ob

print('---------------------------------------------')
print("Metric-level precision on OB:")
print('---------------------------------------------')
print(f'PR@1: {pr_at_1_ob: .2%} ')
print(f'PR@3: {pr_at_3_ob: .2%} ')
print(f'PR@5: {pr_at_5_ob: .2%} ')
print('---------------------------------------------')
print("Metric-level precision on TT:")
print('---------------------------------------------')
print(f'PR@1: {pr_at_1_tt: .2%} ')
print(f'PR@3: {pr_at_3_tt: .2%} ')
print(f'PR@5: {pr_at_5_tt: .2%} ')
print('---------------------------------------------')


