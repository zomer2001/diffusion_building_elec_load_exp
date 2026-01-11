import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 设置后端
matplotlib.use('TkAgg')  # 或者 'MacOSX'

# 从CSV读取数据
file_path = '/Users/zomeryang/Documents/MYSOURCE/diffusion_building_elec_load_exp/similar/cluster_and_train/results/lstm/fixed_training_results_5.csv'  # 替换为你的 CSV 文件路径
df = pd.read_csv(file_path)
print('1111')
# 根据KL_Similarity排序
df_filtered = df.sort_values(by='KL_Similarity')

# 筛选KL_Similarity小于或等于0.5的数据
df_filtered = df_filtered[df_filtered['KL_Similarity']<1]

# 获取所有独立的method
methods = df_filtered['Method'].unique()
#methods = ['diffts','ours','oridata']

# 绘制折线图
plt.figure(figsize=(10, 6))

for method in methods:
    method_data = df_filtered[df_filtered['Method'] == method]
    plt.plot(method_data['KL_Similarity'], method_data['MAE'], label=method, marker='o', markersize=4)  # 设置点的大小为4

# 设置图表标签和标题
plt.title('KL Similarity vs MAE for Different Methods (KL <= 0.5)')
plt.xlabel('KL Similarity')
plt.ylabel('MAE')
plt.legend(title='Method')
plt.grid(True)

# 显示图表
plt.show()
