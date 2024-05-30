import pandas as pd
import os
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
input_folder = 'Data/anomaly log'  # 输入文件夹
output_folder = 'anomaly log classification'  # 输出文件夹
sort_and_save_logs(input_folder, output_folder)
end_time = time.time()
