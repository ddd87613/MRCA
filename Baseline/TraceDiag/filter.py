import pandas as pd
from datetime import datetime, timedelta
import os

def convert_to_datetime(time_str):
    # 从时间字符串中提取日期时间部分，并将其转换为datetime对象
    date_time_str = time_str.split(' +')[0]
    return datetime.fromisoformat(date_time_str)

def filter_data_in_time_range(file_path, start_time, duration_minutes=10):
    # 加载CSV文件
    data = pd.read_csv(file_path)
    # 转换'Time'列为日期时间格式
    data['DateTime'] = data['Time'].apply(convert_to_datetime)

    # 定义目标开始时间和结束时间
    end_time = start_time + timedelta(minutes=duration_minutes)

    # 筛选出目标时间范围内的数据
    filtered_data = data[(data['DateTime'] >= start_time) & (data['DateTime'] <= end_time)]
    return filtered_data

input_folder_path = 'output'
output_folder_path = 'filtered_output'
start_time = datetime(2022, 8, 22, 3, 53)  # 示例时间，需要根据实际情况调整

# 确保输出文件夹存在
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 遍历文件夹中的每个文件
for filename in os.listdir(input_folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder_path, filename)
        filtered_data = filter_data_in_time_range(file_path, start_time)
        # 将筛选出的数据保存到新的CSV文件中
        output_file_path = os.path.join(output_folder_path, filename)
        filtered_data.to_csv(output_file_path, index=False)
        print(f'处理文件：{filename}，筛选出的数据行数：{len(filtered_data)}，保存至：{output_file_path}')
