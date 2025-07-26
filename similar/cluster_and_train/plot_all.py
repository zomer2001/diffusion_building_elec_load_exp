import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import glob
from scipy.signal import savgol_filter

# 设置后端
matplotlib.use('TkAgg')  # 或者 'MacOSX'

# 获取文件夹中所有以"fixed"开头的CSV文件路径
file_path_pattern = '/Users/zomeryang/Documents/bearlab/keti/project/exp/similiar/cluster_and_train/results/lstm/fixed*.csv'
csv_files = glob.glob(file_path_pattern)

# 读取并合并所有CSV文件
df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

# 根据KL_Similarity排序
df_filtered = df.sort_values(by='KL_Similarity')

# 筛选KL_Similarity小于1的数据
df_filtered = df_filtered[df_filtered['KL_Similarity'] < 0.45]

# 获取所有独立的method
methods = df_filtered['Method'].unique()

# 绘制折线图
plt.figure(figsize=(10, 6))
for method in methods:
    method_data = df_filtered[df_filtered['Method'] == method]

    # 对数据进行平滑处理（使用Savitzky-Golay滤波器）
    smoothed_kl_similarity = savgol_filter(method_data['KL_Similarity'], 11, 3)  # 使用窗口大小为11，阶数为3的滤波器
    smoothed_mae = savgol_filter(method_data['MAE'], 11, 3)

    # 绘制平滑后的折线图
    plt.plot(smoothed_kl_similarity, smoothed_mae, label=method, marker='o', markersize=4)  # 设置点的大小为4

# 设置图表标签和标题
plt.title('DIFFERENT SIMILIARITY vs MAE for Different Methods')
plt.xlabel('Similarity')
plt.ylabel('MAE')
plt.legend(title='Method')
plt.grid(True)

# 显示图表
plt.show()
