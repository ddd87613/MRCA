import pandas as pd
import os


def clean_datetime(time_str):
    # 清理时间字符串并转换为datetime对象
    clean_str = time_str.split('+')[0].strip()
    return pd.to_datetime(clean_str, format='%Y-%m-%d %H:%M:%S.%f')


def calculate_deviations(anomaly_data, normal_means):
    # 计算异常数据与正常平均值的偏差
    deviations = anomaly_data[['CpuUsageRate(%)', 'MemoryUsageRate(%)']].sub(normal_means).abs().mean()
    return deviations


# 设置文件夹路径
normal_folder_path = '8-22 metric'
anomaly_folder_path = 'filtered_output'

# 结果列表
results = []

# 遍历异常数据文件夹
for anomaly_filename in os.listdir(anomaly_folder_path):
    if anomaly_filename.endswith('.csv'):
        service_name = anomaly_filename.replace('_metric_anomalies.csv', '')
        anomaly_path = os.path.join(anomaly_folder_path, anomaly_filename)
        normal_path = os.path.join(normal_folder_path, f'{service_name}_metric.csv')

        # 读取异常和正常数据
        if os.path.exists(normal_path):
            data_anomaly = pd.read_csv(anomaly_path)
            data_normal = pd.read_csv(normal_path)

            # 数据清洗和准备
            data_anomaly['DateTime'] = data_anomaly['Time'].apply(clean_datetime)
            data_normal['DateTime'] = data_normal['Time'].apply(clean_datetime)

            # 计算正常数据的平均值
            normal_means = data_normal[['CpuUsageRate(%)', 'MemoryUsageRate(%)']].mean()

            # 计算偏差
            deviations = calculate_deviations(data_anomaly, normal_means)

            # 添加结果
            results.append({
                'Service': service_name,
                'CpuUsageRate_Deviation(%)': deviations['CpuUsageRate(%)'],
                'MemoryUsageRate_Deviation(%)': deviations['MemoryUsageRate(%)'],
                'Total_Deviation': deviations['CpuUsageRate(%)'] + deviations['MemoryUsageRate(%)']
            })

# 创建DataFrame并排序结果
results_df = pd.DataFrame(results)
results_sorted = results_df.sort_values(by='Total_Deviation', ascending=False)
print(results_sorted)
