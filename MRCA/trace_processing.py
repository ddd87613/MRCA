import os
import time
import pandas as pd


def process_trace_files(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有 CSV 文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            data = pd.read_csv(file_path)

            # 通过 SpanID 建立一个索引，便于查找对应的 PodName
            id_to_podname = data.set_index('SpanID')['PodName'].to_dict()

            # 创建一个新列，根据 ParentID 查找对应的 PodName
            data['ParentPodName'] = data['ParentID'].map(id_to_podname)

            # 分组保存文件
            grouped = data.groupby('ParentPodName')
            for pod_name, group in grouped:
                if pd.notna(pod_name):  # 确保 PodName 不是 NaN
                    # 设置 StartTimeUnixNano 为索引，并排序
                    group.set_index('StartTimeUnixNano', inplace=True)
                    group.sort_index(inplace=True)

                    # 只保留 Duration 列
                    output = group[['Duration']]

                    # 保存到文件
                    output_file = os.path.join(output_folder, f"{pod_name}.csv")
                    output.to_csv(output_file)


# 设置输入输出文件夹路径
input_folder = 'raw_data/normal_data/2022-08-22/trace'
output_folder = 'processed_data/normal_data/trace_latency'

# 调用函数处理文件
process_trace_files(input_folder, output_folder)
end_time = time.time()
