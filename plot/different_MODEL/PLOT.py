import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==================== 全局样式设置 ====================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 16,  # 进一步增大基础字体
    'axes.titlesize': 20,  # 增大标题字体
    'axes.labelsize': 18,  # 增大轴标签字体
    'xtick.labelsize': 16,  # 增大x轴刻度标签字体
    'ytick.labelsize': 16,  # 增大y轴刻度标签字体
    'legend.fontsize': 16,  # 增大图例字体
    'figure.dpi': 600,  # 保持高分辨率
    'savefig.dpi': 600,
    'mathtext.fontset': 'stix',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.loc': 'best',
})

# 专业配色方案（保持不变）
MODEL_PALETTE = {
    'RNN': '#4C72B0',  # 蓝色
    'GRU': '#55A868',  # 绿色
    'LSTM': '#C44E52',  # 红色
    'BiLSTM': '#8172B2'  # 紫色
}

# 折线图标记样式（新增：为不同模型设置独特标记）
MODEL_MARKERS = {
    'RNN': 'o',    # 圆形
    'GRU': 's',    # 方形
    'LSTM': '^',   # 三角形
    'BiLSTM': 'D'  # 菱形
}

# ==================== 数据处理 ====================
# 定义CSV文件路径和对应的模型名称（保持不变）
csv_files = {
    'RNN': '../../results/rnn_2160/all_results.csv',
    'GRU': '../../results/gru_2160/all_results.csv',
    'LSTM': '../../results/lstm_2160/all_results.csv',
    'BiLSTM': '../../results/bilstm_2160/all_results.csv'
}

# 方法名称映射（保持不变）
method_mapping = {
    'timegan': 'TIMEGAN',
    'cgan': 'CGAN',
    'diffts': 'DDPM',
    'diffts-fft': 'OURS',
    'wgan': 'VAEGAN',
    'oridata': 'Real Data'
}

# 存储所有模型的数据（保持不变）
all_data = []

# 读取并处理每个CSV文件（保持不变）
for model_name, file_path in csv_files.items():
    try:
        df = pd.read_csv(file_path)
        df['Method'] = df['Method'].map(method_mapping)
        df_70 = df[df['Sparsity'] == 70]
        mae_means = df_70.groupby('Method')['MAE'].mean().reset_index()
        mae_means['Model'] = model_name
        all_data.append(mae_means)
    except Exception as e:
        print(f"Error processing {model_name} data: {str(e)}")

# 合并所有数据（保持不变）
combined_df = pd.concat(all_data, ignore_index=True)
method_order = ['TIMEGAN', 'CGAN', 'DDPM', 'VAEGAN', 'Real Data', 'OURS']
combined_df['Method'] = pd.Categorical(combined_df['Method'], categories=method_order, ordered=True)
combined_df = combined_df.sort_values('Method')

# ==================== 折线图绘制 ====================
plt.figure(figsize=(12, 8), tight_layout=True)
ax = plt.gca()

# 绘制折线图（调整标记样式和线宽）
sns.lineplot(
    data=combined_df,
    x='Method',
    y='MAE',
    hue='Model',
    style='Model',
    markers=MODEL_MARKERS,  # 使用自定义标记
    dashes=False,
    markersize=10,  # 增大标记大小
    linewidth=4,    # 进一步加粗线条
    palette=MODEL_PALETTE,
    ax=ax
)

# 设置标题和标签（保持字体增大）
plt.title('MAE Comparison Across Different Models (70% Training Data)',
          fontsize=22, pad=15, weight='bold')
plt.xlabel('Generation Method', weight='bold', fontsize=20)
plt.ylabel('MAE', weight='bold', fontsize=20)

# 优化图例（保持原位置和样式）
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles, labels,
    title='Prediction Model',
    title_fontsize='16',
    frameon=True,
    shadow=True,
    fancybox=True,
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=16
)

# 添加数据标签（增大字体）
for model in MODEL_PALETTE.keys():
    model_data = combined_df[combined_df['Model'] == model]
    for i, row in model_data.iterrows():
        ax.text(row['Method'], row['MAE'], f"{row['MAE']:.3f}",
                color=MODEL_PALETTE[model],
                fontsize=16, ha='center', va='bottom', weight='bold')  # 增大标签字体

# 添加网格（保持不变）
plt.grid(True, linestyle=':', alpha=0.5)

# 调整x轴标签：不旋转 + 增大字体
plt.xticks(rotation=0, fontsize=18)  # 明确设置不旋转

# 设置y轴刻度标签字体大小
plt.yticks(fontsize=18)

# 保存结果（保持不变）
plt.savefig('mae_comparison_across_models.pdf', bbox_inches='tight', dpi=600)
plt.savefig('mae_comparison_across_models.png', bbox_inches='tight', dpi=600)
print("已保存折线图：mae_comparison_across_models.pdf 和 mae_comparison_across_models.png")

# 显示处理后的数据（保持不变）
print("\n=== 处理后数据 ===")
print(combined_df)