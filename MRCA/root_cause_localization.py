import glob
import json
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import os
import pandas as pd



def causal_anlysis(services_folder, front_service_path,times_file,ground_truth_path,xxx):
    # 筛选需要的列
    required_columns = ['CpuUsageRate(%)', 'PodClientLatencyP99(s)', 'NodeNetworkReceiveBytes']

    # 加载 Ground Truth 数据
    with open(ground_truth_path, 'r') as file:
        ground_truth_data = json.load(file)

    # 将 Ground Truth 数据转换为 DataFrame 格式
    ground_truth = []
    for hour, entries in ground_truth_data.items():
        for entry in entries:
            ground_truth.append({
                'InjectionTime': pd.to_datetime(pd.to_numeric(entry['inject_timestamp']), unit='s'),
                'ServiceName': entry['inject_pod'].split('-')[0],
                'InjectType': entry['inject_type']
            })

    ground_truth_df = pd.DataFrame(ground_truth)

    def filter_and_save_services(input_folder, required_columns, window_start, window_end):
        all_files = glob.glob(os.path.join(input_folder, '*.csv'))
        services_data = {}

        for file in all_files:
            service_name = os.path.splitext(os.path.basename(file))[0]
            data = pd.read_csv(file)

            if 'Time' in data.columns:
                data['Time'] = data['Time'].apply(lambda x: x.split(' +')[0])
                data['Time'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S.%f')
                data = data.loc[(data['Time'] >= window_start) & (data['Time'] <= window_end)]
                data.set_index('Time', inplace=True)
                if all(col in data.columns for col in required_columns):
                    filtered_data = data[required_columns]
                    filtered_data.replace([np.inf, -np.inf], np.nan, inplace=True)
                    filtered_data.dropna(inplace=True)
                    services_data[service_name] = filtered_data

        return services_data

    def load_and_filter_front_service(front_service_path, window_start, window_end):
        front_service_data = pd.read_csv(front_service_path)

        front_service_data['Time'] = front_service_data['Time'].apply(lambda x: x.split(' +')[0])
        front_service_data['Time'] = pd.to_datetime(front_service_data['Time'], format='%Y-%m-%d %H:%M:%S.%f')
        front_service_data.set_index('Time', inplace=True)

        # 选择需要的指标并筛选时间窗口
        front_service_metrics = ['SuccessRate(%)', 'LatencyP50(s)', 'LatencyP90(s)', 'LatencyP95(s)', 'LatencyP99(s)']
        return front_service_data[front_service_metrics].loc[window_start:window_end]

    def perform_causality_test(df, front_service_metrics, other_service_metrics, max_lag=2):
        results = []
        for front_metric in front_service_metrics:
            for other_metric in other_service_metrics:
                try:
                    subset = df[[front_metric, other_metric]].dropna()
                    if len(subset) < max_lag + 1:
                        continue
                    causality_result = grangercausalitytests(subset, max_lag, verbose=False)
                    p_value = min([causality_result[lag][0]['ssr_ftest'][1] for lag in causality_result])
                    results.append((front_metric, other_metric, p_value))
                except Exception as e:
                    print(f"Skipping metric {other_metric} due to error: {e}")
        return sorted(results, key=lambda x: x[2], reverse=True)

    def find_causality_between_services(front_service_data, services_data, front_service_metrics, max_lag=2):
        results = []
        for service_name, service_data in services_data.items():
            numeric_service_data = service_data.select_dtypes(include='number')
            merged_data = pd.merge_asof(front_service_data.sort_index(), numeric_service_data.sort_index(), on='Time')
            pc_results = perform_causality_test(merged_data, front_service_metrics, numeric_service_data.columns,
                                                     max_lag)
            results.extend(
                [(service_name, front_metric, metric, p_value) for front_metric, metric, p_value in pc_results])
        return results

    def load_injection_times(times_file):
        with open(times_file, 'r') as f:
            times = [line.strip() for line in f.readlines() if line.strip()]
        return [pd.to_datetime(time) for time in times]

    injection_times = load_injection_times(times_file)

    tr = []


    for target_timestamp in injection_times:
        window_start = target_timestamp - pd.Timedelta(minutes=5)
        window_end = target_timestamp + pd.Timedelta(minutes=5)

        front_service_data = load_and_filter_front_service(front_service_path, window_start, window_end)
        services_data = filter_and_save_services(services_folder, required_columns, window_start, window_end)

        results = find_causality_between_services(front_service_data, services_data, front_service_data.columns,
                                                  max_lag=2)
        results = sorted(results, key=lambda x: x[3], reverse=True)[:5]

        for service_name, front_metric, other_metric, p_value in results:
            tr.append((target_timestamp, service_name, front_metric, other_metric, p_value))
    tr_df = pd.DataFrame(tr, columns=['InjectionTime', 'ServiceName', 'FrontMetric', 'OtherMetric', 'PValue'])
    grouped_predictions = tr_df.groupby('InjectionTime')
    tr_df.to_csv(xxx)


causal_anlysis('raw_data/abnormal/suffer_anomaly_inject_data_1/metric', 'raw_data/abnormal/suffer_anomaly_inject_data_1/front_service.csv',
                                  'raw_data/abnormal/suffer_anomaly_inject_data_1/inject anomaly time point.txt',
                                  'raw_data/abnormal/suffer_anomaly_inject_data_1/2022-08-22-fault_list.json','RCA/result-8-22.csv')
causal_anlysis('raw_data/abnormal/suffer_anomaly_inject_data_2/metric', 'raw_data/abnormal/suffer_anomaly_inject_data_2/front_service.csv', 'raw_data/abnormal/suffer_anomaly_inject_data_2/inject anomaly time point.txt', 'raw_data/abnormal/suffer_anomaly_inject_data_2/2022-08-23-fault_list.json','RCA/result-8-23.csv')
causal_anlysis('raw_data/abnormal/suffer_anomaly_inject_data_3/metric', 'raw_data/abnormal/suffer_anomaly_inject_data_3/front_service.csv',
                                  'raw_data/abnormal/suffer_anomaly_inject_data_3/time.txt',
                                  'raw_data/abnormal/suffer_anomaly_inject_data_3/2023-01-29-fault_list.json', 'RCA/result-1-29.csv')
causal_anlysis('raw_data/abnormal/suffer_anomaly_inject_data_4/metric', 'raw_data/abnormal/suffer_anomaly_inject_data_4/front_service.csv',
                                  'raw_data/abnormal/suffer_anomaly_inject_data_4/time.txt',
                                  'raw_data/abnormal/suffer_anomaly_inject_data_4/2023-01-30-fault_list.json', 'RCA/result-1-30.csv')

def rename_and_modify_metrics(experiment_results):
    def classify_metric(metric):
        if 'Latency' in metric or 'NodeNetwork' in metric:
            return 'network_delay'
        elif 'Cpu' in metric:
            return 'cpu_contention'
        else:
            return metric

    if 'PValue' in experiment_results.columns:
        experiment_results = experiment_results.drop(columns=['PValue'])
    if 'FrontMetric' in experiment_results.columns:
        experiment_results = experiment_results.drop(columns=['FrontMetric'])

    experiment_results['anomaly_type'] = experiment_results['OtherMetric'].apply(classify_metric)
    experiment_results = experiment_results.drop(columns=['OtherMetric'])

    return experiment_results

def prune_causal_graph(causal_graph, ranking_list):
    pruned_ranking_list = ranking_list.copy()

    # Iterate through the causal graph
    for node, parents in causal_graph.items():
        # Check if the node has parents and if the node is in the ranking list
        if parents and node in pruned_ranking_list:
            # Remove the node from the ranking list
            pruned_ranking_list.remove(node)

    return pruned_ranking_list


# 加载新的实验结果文件
experiment_results1 = pd.read_csv('RCA/result-8-22.csv')
experiment_results2 = pd.read_csv('RCA/result-8-23.csv')
experiment_results3 = pd.read_csv('RCA/result-1-29.csv')
experiment_results4 = pd.read_csv('RCA/result-1-30.csv')

# 应用重命名和修改函数
experiment_results1_modified = rename_and_modify_metrics(experiment_results1)
experiment_results2_modified = rename_and_modify_metrics(experiment_results2)
experiment_results3_modified = rename_and_modify_metrics(experiment_results3)
experiment_results4_modified = rename_and_modify_metrics(experiment_results4)

# 保存修改后的DataFrame到新的CSV文件
output_path1 = 'RCA/result-8-22.csv'
output_path2 = 'RCA/result-8-23.csv'
output_path3 = 'RCA/result-1-29.csv'
output_path4 = 'RCA/result-1-30.csv'
experiment_results1_modified.to_csv(output_path1, index=False)
experiment_results2_modified.to_csv(output_path2, index=False)
experiment_results3_modified.to_csv(output_path3, index=False)
experiment_results4_modified.to_csv(output_path4, index=False)
