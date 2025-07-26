import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

def sliding_window(data, window_size=24):
    """
    使用滑动窗口生成数据样本
    参数：
        data: 输入的二维数据数组，形状为 [n_samples, n_features]
        window_size: 窗口大小（默认为24）
    返回：
        通过滑动窗口生成的三维数组，每个样本的形状为 [window_size, n_features]
    """
    samples = []
    for i in range(len(data) - window_size + 1):
        samples.append(data[i:i + window_size])  # 滑动窗口提取样本
    return np.array(samples)

def save_as_npy(file_path, data, output_dir, i):
    """
    将数据保存为npy文件
    参数：
        file_path: 输入文件的路径
        data: 要保存的数据（需要为二维数组）
        output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 构造输出文件路径
    file_name = os.path.splitext(os.path.basename(file_path))[0] + '.npy'
    npy_file_path = os.path.join(output_dir, file_name)

    np.save(f'{npy_file_path}_{i}', data)  # 加上索引以避免文件覆盖
    print(f"Saved {file_name} to {npy_file_path}_{i}.npy")

# 设置输入文件路径和输出目录
building_test_name = 'Bobcat_warehouse_Charlie'
csv_file_path = '/Users/zomeryang/Documents/bearlab/keti/project/data_prepare/standard_selected/Bobcat_warehouse_Charlie.csv'  # 替换为CSV文件路径
output_dir = './output_npy_files'  # 输出目录

# 从CSV文件读取数据
df = pd.read_csv(csv_file_path)

# 假设有多个特征，我们需要使用所有特征（例如：负荷、气温等）
# 假设 CSV 文件中前 6 列是我们需要的特征
data = df.iloc[:, :6].values  # 获取前6列数据，形状 [n_samples, 6]

# 归一化处理：使用MinMaxScaler对每一列进行归一化
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)  # 归一化数据

# 使用滑动窗口将数据转换为24步的样本
window_size = 24
sliding_data = sliding_window(data_normalized, window_size)

# 每个npy文件保存2160个样本（如果数据量足够的话）
n_samples_per_npy = 1080
num_files = len(sliding_data) // n_samples_per_npy  # 确保样本数可以被2160整除

for i in range(num_files):
    start_idx = i * n_samples_per_npy
    end_idx = start_idx + n_samples_per_npy
    samples = sliding_data[start_idx:end_idx]  # 获取每2160个样本

    # 保存为npy文件
    save_as_npy(csv_file_path, samples, output_dir, i)

print(f"Total {num_files} .npy files have been saved.")
