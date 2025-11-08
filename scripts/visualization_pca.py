import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==================== 全局样式设置 ====================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'mathtext.fontset': 'stix',
    'axes.grid': False,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.loc': 'best',
    'axes.linewidth': 0.8,
    'axes.edgecolor': 'black'
})

# 配色方案
COLOR_PALETTE = {
    'oridata': '#1f77b4',  # 蓝色 - 原始数据
    'testdata': '#d62728',  # 红色 - 测试数据
    'DDPM': '#ff7f0e',  # 橙色 - Ours方法(diffts-fft)
    'OURS': '#2ca02c'  # 绿色 - DDPM方法(diffts)
}

# 只处理2160长度和70稀疏率
target_length = 2160
target_sparsity = 70
base_dir = '../fakedata'
test_data_folder = '../testdata'
output_dir = '../results/pca'
os.makedirs(output_dir, exist_ok=True)


def prepare_pca_data(data, max_samples=None):
    """准备用于PCA分析的数据"""
    if data is None or len(data) == 0:
        return None

    if max_samples is None or len(data) <= max_samples:
        return data.reshape(data.shape[0], -1)

    indices = np.random.choice(len(data), max_samples, replace=False)
    sampled_data = data[indices]
    return sampled_data.reshape(sampled_data.shape[0], -1)


def load_generated_data(folder_path):
    """加载生成的合成数据"""
    data = []
    if os.path.exists(folder_path):
        for sub_folder in os.listdir(folder_path):
            sub_folder_path = os.path.join(folder_path, sub_folder)
            if os.path.isdir(sub_folder_path):
                for file in os.listdir(sub_folder_path):
                    if file.endswith('.npy'):
                        file_path = os.path.join(sub_folder_path, file)
                        try:
                            data.append(np.load(file_path))
                        except Exception as e:
                            print(f"加载文件失败: {file_path}, 错误: {str(e)}")
        if data:
            return np.concatenate(data, axis=0)
    return None


def plot_pca_analysis(data_dict, building_name, sparsity):
    """绘制PCA分析图"""
    # 准备数据
    datasets = []
    data_types = []
    sample_counts = []

    for data_type in ['oridata', 'testdata', 'DDPM', 'OURS']:
        if data_type in data_dict and data_dict[data_type] is not None:
            # 对生成数据进行采样以避免过载
            max_samples = 500 if data_type in ['DDPM', 'OURS'] else None
            prepared_data = prepare_pca_data(data_dict[data_type], max_samples)
            if prepared_data is not None:
                datasets.append(prepared_data)
                data_types.append(data_type)
                sample_counts.append(prepared_data.shape[0])

    if len(datasets) < 2:
        print(f"数据不足，无法为{building_name}_{sparsity}生成PCA图")
        return

    # 合并和标准化数据
    combined_data = np.vstack(datasets)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)

    # 执行PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(scaled_data)

    # 计算方差解释比例
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.sum()

    # 创建绘图数据框
    labels = []
    for i, data_type in enumerate(data_types):
        labels.extend([data_type] * sample_counts[i])

    pca_df = pd.DataFrame({
        'PC1': pca_results[:, 0],
        'PC2': pca_results[:, 1],
        'Data Type': labels
    })

    # 创建子图布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), tight_layout=True)

    # 子图1: PCA散点图
    sns.scatterplot(
        x='PC1',
        y='PC2',
        hue='Data Type',
        style='Data Type',
        data=pca_df,
        palette=[COLOR_PALETTE[dt] for dt in pca_df['Data Type'].unique()],
        s=60,
        alpha=0.7,
        ax=ax1,
        markers={'oridata': 'o', 'testdata': 's', 'OURS': 'D', 'DDPM': '^'}
    )

    ax1.set_xlabel(f'Principal Component 1 ({explained_variance[0]:.2%})',
                   fontsize=12, weight='bold')
    ax1.set_ylabel(f'Principal Component 2 ({explained_variance[1]:.2%})',
                   fontsize=12, weight='bold')
    ax1.set_title(f'PCA Projection - {building_name}\n'
                  f'Total Variance Explained: {cumulative_variance:.2%}',
                  fontsize=13, weight='bold', pad=15)

    # 子图2: 方差解释比例
    components = range(1, min(10, len(pca.explained_variance_ratio_)) + 1)
    variance_ratios = pca.explained_variance_ratio_[:len(components)]
    cumulative = np.cumsum(variance_ratios)

    ax2.bar(components, variance_ratios, alpha=0.7,
            color='skyblue', label='Individual')
    ax2.plot(components, cumulative, marker='o', color='red',
             linewidth=2, markersize=6, label='Cumulative')

    ax2.set_xlabel('Principal Components', fontsize=12, weight='bold')
    ax2.set_ylabel('Explained Variance Ratio', fontsize=12, weight='bold')
    ax2.set_title('PCA Variance Explained', fontsize=13, weight='bold', pad=15)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 设置整体标题
    fig.suptitle(f'PCA Analysis - Building: {building_name} ({sparsity}% Sparsity)',
                 fontsize=14, weight='bold', y=0.98)

    # 美化图例
    legend = ax1.legend(
        title='Data Type',
        title_fontsize=11,
        fontsize=10,
        loc='best',
        frameon=True,
        framealpha=0.9,
        edgecolor='black',
        markerscale=1.2
    )

    # 保存结果
    filename = f"{building_name.replace(' ', '_').replace('/', '_')}_sparsity_{sparsity}_pca.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"已保存PCA图: {output_path}")

    # 打印PCA信息
    print(f"PCA方差解释比例: {explained_variance}")
    print(f"累计方差解释: {cumulative_variance:.2%}")

    plt.close()


# 主逻辑
found_target = False

for test_folder in os.listdir(test_data_folder):
    if found_target:
        break

    parts = test_folder.split('_')
    if len(parts) >= 4:
        building_name = '_'.join(parts[:-2])  # 修正建筑名提取
        length = int(parts[-2])
        sparsity = int(parts[-1])

        # 只处理目标长度和稀疏率
        if length == target_length and sparsity == target_sparsity:
            found_target = True

            print(f"处理建筑: {building_name}, 长度: {length}, 稀疏率: {sparsity}%")

            # 读取原始数据和测试数据
            oridata_file = os.path.join(test_data_folder, test_folder, 'samples', 'energy_norm_truth_24_train.npy')
            test_file = os.path.join(test_data_folder, test_folder, 'samples', 'energy_norm_truth_24_test.npy')

            if not os.path.exists(oridata_file) or not os.path.exists(test_file):
                print(f"原始数据或测试数据文件不存在, 跳过...")
                continue

            oridata = np.load(oridata_file)
            test_data = np.load(test_file)

            # 加载DDPM (diffts) 数据
            ddpm_folder = os.path.join(base_dir, 'diffts', str(sparsity), building_name)
            ddpm_data = load_generated_data(ddpm_folder)
            print(f"加载了 {len(ddpm_data) if ddpm_data is not None else 0} 个DDPM生成的样本")

            # 加载OURS (diffts-fft) 数据
            ours_folder = os.path.join(base_dir, 'diffts-fft', str(sparsity), building_name)
            ours_data = load_generated_data(ours_folder)
            print(f"加载了 {len(ours_data) if ours_data is not None else 0} 个OURS生成的样本")

            # 创建数据字典
            data_dict = {
                'oridata': oridata,
                'testdata': test_data,
                'DDPM': ddpm_data,
                'OURS': ours_data
            }

            # 生成并保存PCA图
            plot_pca_analysis(data_dict, building_name, sparsity)
            break

if not found_target:
    print(f"未找到长度为{target_length}、稀疏率为{target_sparsity}%的数据")

print("PCA分析完成！")