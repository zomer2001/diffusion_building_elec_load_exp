import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==================== 全局样式设置 ====================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 14,  # 增大基础字体大小
    'axes.titlesize': 16,  # 增大标题字体
    'axes.labelsize': 15,  # 增大轴标签字体
    'xtick.labelsize': 13,  # 增大x轴刻度标签字体
    'ytick.labelsize': 13,  # 增大y轴刻度标签字体
    'legend.fontsize': 13,  # 增大图例字体
    'figure.dpi': 600,  # 提高分辨率
    'savefig.dpi': 600,  # 提高保存分辨率
    'mathtext.fontset': 'stix',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.loc': 'best',
})

# 专业配色方案
MODEL_PALETTE = {
    'RNN': '#4C72B0',  # 蓝色
    'GRU': '#55A868',  # 绿色
    'LSTM': '#C44E52',  # 红色
    'BiLSTM': '#8172B2'  # 紫色
}

# ==================== 数据处理 ====================
# 定义CSV文件路径和对应的模型名称
csv_files = {
    'RNN': '../../results/rnn_2160/all_results.csv',
    'GRU': '../../results/gru_2160/all_results.csv',
    'LSTM': '../../results/lstm_2160/all_results.csv',
    'BiLSTM': '../../results/bilstm_2160/all_results.csv'
}

# 方法名称映射
method_mapping = {
    'timegan': 'TIMEGAN',
    'cgan': 'CGAN',
    'diffts': 'DDPM',
    'diffts-fft': 'OURS',
    'wgan': 'VAEGAN',
    'oridata': 'Real Data'
}

# 存储所有模型的数据
all_data = []

# 读取并处理每个CSV文件
for model_name, file_path in csv_files.items():
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 应用方法名称映射
        df['Method'] = df['Method'].map(method_mapping)

        # 筛选Sparsity=70的数据
        df_70 = df[df['Sparsity'] == 70]

        # 按方法分组计算MAE平均值
        mae_means = df_70.groupby('Method')['MAE'].mean().reset_index()
        mae_means['Model'] = model_name  # 添加模型名称列

        all_data.append(mae_means)
    except Exception as e:
        print(f"Error processing {model_name} data: {str(e)}")

# 合并所有数据
combined_df = pd.concat(all_data, ignore_index=True)

# 确保方法顺序正确
method_order = ['TIMEGAN', 'CGAN', 'DDPM', 'VAEGAN', 'Real Data', 'OURS']
combined_df['Method'] = pd.Categorical(combined_df['Method'], categories=method_order, ordered=True)
combined_df = combined_df.sort_values('Method')

# ==================== 折线图绘制 ====================
plt.figure(figsize=(12, 8), tight_layout=True)  # 增大图形尺寸
ax = plt.gca()

# 绘制折线图
sns.lineplot(
    data=combined_df,
    x='Method',
    y='MAE',
    hue='Model',
    style='Model',
    markers=True,
    dashes=False,
    markersize=16,  # 增大标记大小
    linewidth=3,    # 增大线宽
    palette=MODEL_PALETTE,
    ax=ax
)

# 设置标题和标签
plt.title('MAE Comparison Across Different Models (70% Training Data)',
          fontsize=20, pad=15, weight='bold')  # 增大标题字体
plt.xlabel('Generation Method', weight='bold', fontsize=18)  # 增大标签字体
plt.ylabel('MAE', weight='bold', fontsize=18)  # 增大标签字体

# 优化图例
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles, labels,
    title='Prediction Model',
    title_fontsize='14',  # 增大图例标题字体
    frameon=True,
    shadow=True,
    fancybox=True,
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=16  # 增大图例字体
)

# 添加数据标签
for model in MODEL_PALETTE.keys():
    model_data = combined_df[combined_df['Model'] == model]
    for i, row in model_data.iterrows():
        ax.text(row['Method'], row['MAE'], f"{row['MAE']:.3f}",
                color=MODEL_PALETTE[model],
                fontsize=14, ha='center', va='bottom', weight='bold')  # 增大标签字体

# 添加网格
plt.grid(True, linestyle=':', alpha=0.5)

# 调整x轴标签角度
plt.xticks( fontsize=18)  # 增大x轴标签字体

# 设置y轴刻度标签字体大小
plt.yticks(fontsize=18)

# 保存结果
plt.savefig('mae_comparison_across_models.pdf', bbox_inches='tight', dpi=600)
plt.savefig('mae_comparison_across_models.png', bbox_inches='tight', dpi=600)
print("已保存折线图：mae_comparison_across_models.pdf 和 mae_comparison_across_models.png")

# 显示处理后的数据
print("\n=== 处理后数据 ===")
print(combined_df)