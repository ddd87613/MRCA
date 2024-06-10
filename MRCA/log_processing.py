import os
import json
import pandas as pd
from drain3 import TemplateMiner
from pathlib import Path
import time

star_time = time.time()
def sort_and_save_logs(input_folder, output_folder):
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)

    # 创建一个空的DataFrame用于存放所有文件数据
    all_data = pd.DataFrame()

    # 遍历文件夹中的所有文件，合并到一个DataFrame
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            csv_path = os.path.join(input_folder, filename)
            data = pd.read_csv(csv_path)
            all_data = pd.concat([all_data, data], ignore_index=True)

    # 如果有Timestamp列，将其转换为datetime类型以便排序
    if 'Timestamp' in all_data.columns:
        all_data['Timestamp'] = pd.to_datetime(all_data['Timestamp'])

    # 按照'PodName'进行分类，并按时间排序后保存每个类别到新的CSV文件中
    for pod_name, group in all_data.groupby('PodName'):
        sorted_group = group.sort_values('Timestamp')  # 按时间排序

        # 生成安全的文件名
        safe_filename = f"{pod_name.replace('-', '_')}.csv"
        file_path = os.path.join(output_folder, safe_filename)

        # 保存数据到新的CSV文件
        sorted_group.to_csv(file_path, index=False)

        print(f"All data for PodName {pod_name} saved in {file_path}")

# 调用函数
input_folder = 'raw_data/normal_data/2022-08-22/log'  # 输入文件夹
output_folder = 'processed_data/anomaly log classification'  # 输出文件夹
sort_and_save_logs(input_folder, output_folder)
end_time = time.time()

def parse_log(log):
    try:
        outer_json = json.loads(log)
        if isinstance(outer_json['log'], str):
            return outer_json['log']
        else:
            inner_json = json.loads(outer_json['log'])
            return inner_json['message']
    except json.JSONDecodeError as e:
        print("JSON Decode Error in log:", log)
        raise e
    except Exception as e:
        print("Error processing log:", log)
        raise e

def process_log_file(log_path, output_dir):
    log_data = pd.read_csv(log_path)

    log_data['Log'] = log_data['Log'].apply(parse_log)

    template_miner = TemplateMiner()

    log_data['template_id'] = log_data['Log'].apply(lambda log_message: template_miner.add_log_message(log_message)['cluster_id'])

    log_data['Timestamp'] = pd.to_datetime(log_data['Timestamp'])
    log_data.set_index('Timestamp', inplace=True)

    frequency = log_data.groupby('template_id').resample('5S').size().unstack(level=0, fill_value=0)

    threshold = 0.8 * len(frequency)
    frequency = frequency.loc[:, (frequency == 0).sum(axis=0) < threshold]

    frequency = frequency.loc[~(frequency == 0).all(axis=1)]

    filename = Path(log_path).stem + '_frequency.csv'  # Creates a new filename based on the original
    frequency.to_csv(os.path.join(output_dir, filename))

# Input and output directories
input_dir = 'processed_data/anomaly log classification'
output_dir = 'processed_data/normal_data/log_template'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each file in the directory
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        log_path = os.path.join(input_dir, filename)
        process_log_file(log_path, output_dir)
