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

# 定义稀疏率和数据长度
sparsity_rates = [90, 70, 50, 30]
base_dir = '../fakedata'
test_data_folder = '../testdata'
output_dir = '../results/pca/ddpm_and_ours'  # 修改为PCA输出目录
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


def plot_pca(data_dict, building_name, sparsity):
    """绘制高质量的PCA分布图"""
    # 准备数据
    datasets = []
    data_types = []

    for data_type in ['oridata', 'testdata', 'DDPM', 'OURS']:
        if data_type in data_dict and data_dict[data_type] is not None:
            max_samples = 500 if data_type in ['DDPM', 'OURS'] else None
            prepared_data = prepare_pca_data(data_dict[data_type], max_samples)
            if prepared_data is not None:
                datasets.append(prepared_data)
                data_types.append(data_type)

    if len(datasets) < 2:
        print(f"数据不足，无法为{building_name}_{sparsity}生成PCA图")
        return

    # 合并和标准化数据
    combined_data = np.vstack(datasets)
    scaled_data = StandardScaler().fit_transform(combined_data)

    # 运行PCA
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(scaled_data)

    # 计算方差解释比例
    explained_variance = pca.explained_variance_ratio_
    total_variance = explained_variance.sum()

    # 创建绘图数据框
    labels = []
    for i, data_type in enumerate(data_types):
        labels.extend([data_type] * datasets[i].shape[0])

    pca_df = pd.DataFrame({
        'PC1': pca_results[:, 0],
        'PC2': pca_results[:, 1],
        'Data Type': labels
    })

    # 绘制PCA图
    plt.figure(figsize=(10, 8), tight_layout=True)
    ax = plt.gca()

    sns.scatterplot(
        x='PC1',
        y='PC2',
        hue='Data Type',
        style='Data Type',
        data=pca_df,
        palette=[COLOR_PALETTE[dt] for dt in pca_df['Data Type'].unique()],
        s=80,
        alpha=0.8,
        ax=ax,
        markers={'oridata': 'o', 'testdata': 's', 'OURS': 'D', 'DDPM': '^'}
    )

    # 设置标题和标签
    plt.title(f'PCA Distribution - Building: {building_name} ({sparsity}% Sparsity)\n'
              f'Explained Variance: PC1={explained_variance[0]:.1%}, PC2={explained_variance[1]:.1%}, Total={total_variance:.1%}',
              fontsize=14, pad=15, weight='bold')
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.1%} variance)', fontsize=12, weight='bold')
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.1%} variance)', fontsize=12, weight='bold')

    # 美化图例
    legend = ax.legend(
        title='Data Type',
        title_fontsize=11,
        fontsize=10,
        loc='best',
        frameon=True,
        framealpha=0.9,
        edgecolor='black',
        markerscale=1.5
    )

    # 保存结果
    filename = f"{building_name.replace(' ', '_').replace('/', '_')}_sparsity_{sparsity}_pca.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"已保存PCA图: {output_path}")
    plt.close()


# 主逻辑
processed_buildings = set()

for test_folder in os.listdir(test_data_folder):
    parts = test_folder.split('_')
    if len(parts) >= 4:
        building_name = '_'.join(parts[:-1])
        length = int(parts[-2])
        sparsity = int(parts[-1])

        key = f"{building_name}_{sparsity}"
        if key in processed_buildings or length != 2160 or sparsity not in sparsity_rates:
            continue

        processed_buildings.add(key)

        print(f"处理建筑: {building_name}, 稀疏率: {sparsity}%")

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
        plot_pca(data_dict, building_name, sparsity)

print("所有PCA图生成完成！")