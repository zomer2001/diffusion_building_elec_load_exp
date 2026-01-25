import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==================== 全局样式设置 ====================
# 核心原则：使用有顺序的字体列表，确保中英文都能正常显示
plt.rcParams.update({
    'font.family': ['Times New Roman', 'SimSun', 'Microsoft YaHei'],
    'axes.unicode_minus': False,

    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,

    'figure.dpi': 600,
    'savefig.dpi': 600,
    'mathtext.fontset': 'stix',

    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.loc': 'best',
})

# ==================== 配色与样式 ====================
MODEL_PALETTE = {
    'RNN': '#4C72B0',
    'GRU': '#55A868',
    'LSTM': '#C44E52',
    'BiLSTM': '#8172B2'
}

MODEL_MARKERS = {
    'RNN': 'o',
    'GRU': 's',
    'LSTM': '^',
    'BiLSTM': 'D'
}

# ==================== 数据处理 ====================
csv_files = {
    'RNN': '../../results/rnn_2160/all_results.csv',
    'GRU': '../../results/gru_2160/all_results.csv',
    'LSTM': '../../results/lstm_2160/all_results.csv',
    'BiLSTM': '../../results/bilstm_2160/all_results.csv'
}

method_mapping = {
    'timegan': 'TIMEGAN',
    'cgan': 'CGAN',
    'diffts': 'DDPM',
    'diffts-fft': 'OURS',
    'wgan': 'VAEGAN',
    'oridata': 'Real Data'
}

all_data = []

for model_name, file_path in csv_files.items():
    try:
        df = pd.read_csv(file_path)
        df['Method'] = df['Method'].map(method_mapping)

        df_70 = df[df['Sparsity'] == 70]
        mae_means = df_70.groupby('Method')['MAE'].mean().reset_index()
        mae_means['Model'] = model_name

        all_data.append(mae_means)
    except Exception as e:
        print(f"{model_name} 数据处理失败: {str(e)}")

combined_df = pd.concat(all_data, ignore_index=True)

method_order = ['TIMEGAN', 'CGAN', 'DDPM', 'VAEGAN', 'Real Data', 'OURS']
combined_df['Method'] = pd.Categorical(
    combined_df['Method'],
    categories=method_order,
    ordered=True
)
combined_df = combined_df.sort_values('Method')

# ==================== 折线图绘制 ====================
plt.figure(figsize=(16, 6), tight_layout=True)
ax = plt.gca()

sns.lineplot(
    data=combined_df,
    x='Method',
    y='MAE',
    hue='Model',
    style='Model',
    markers=MODEL_MARKERS,
    dashes=False,
    markersize=10,
    linewidth=4,
    palette=MODEL_PALETTE,
    ax=ax
)

# ==================== 中文标题与坐标轴 ====================
plt.title(
    '不同预测模型在 70% 训练数据条件下的 MAE 对比',
    fontsize=22, pad=15, weight='bold'
)
plt.xlabel('数据生成方法', weight='bold', fontsize=20)
plt.ylabel('MAE', weight='bold', fontsize=20)

# ==================== 图例设置 ====================
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles, labels,
    title='预测模型',
    title_fontsize=16,
    frameon=True,
    shadow=True,
    fancybox=True,
    loc='upper right',
    fontsize=16
)

# ==================== 数值标注 ====================
for model in MODEL_PALETTE.keys():
    model_data = combined_df[combined_df['Model'] == model]
    for _, row in model_data.iterrows():
        ax.text(
            row['Method'],
            row['MAE'],
            f"{row['MAE']:.3f}",
            color=MODEL_PALETTE[model],
            fontsize=16,
            ha='center',
            va='bottom',
            weight='bold'
        )

# ==================== 细节优化 ====================
plt.grid(True, linestyle=':', alpha=0.5)
plt.xticks(rotation=0, fontsize=18)
plt.yticks(fontsize=18)

# ==================== 保存结果 ====================
plt.savefig('mae_comparison_across_models.pdf', bbox_inches='tight', dpi=600)
plt.savefig('mae_comparison_across_models.png', bbox_inches='tight', dpi=600)

print("已保存折线图：mae_comparison_across_models.pdf 和 mae_comparison_across_models.png")

print("\n=== 处理后数据 ===")
print(combined_df)
