import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ==================== 全局样式设置（字体增大）====================
plt.rcParams.update({
    'font.family': ['Times New Roman', 'SimSun', 'Microsoft YaHei'],
    'axes.unicode_minus': False,

    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,

    'figure.dpi': 1200,
    'savefig.dpi': 1200,
    'mathtext.fontset': 'stix',

    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.4,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.loc': 'best',
})
# 专业学术配色方案
PALETTE = {
    'bar': '#2c7bb6',
    'line': '#d7191c',
    'marker': '#fdae61',
    'text': '#2c3e50',
    'background': '#ffffff',
    'grid': '#e0e0e0'
}

# 配色方案（保持不变）
COLOR_PALETTE = {
    'Traindata': '#1f77b4',
    'Testdata': '#d62728',
    'DDPM': '#ff7f0e',
    'OURS': '#2ca02c'
}

# 定义稀疏率和路径（保持不变）
sparsity_rates = [70]
base_dir = '../fakedata'
test_data_folder = '../testdata'
output_dir = '../results/tsne/ddpm_and_ours_lunwen1'
os.makedirs(output_dir, exist_ok=True)


def prepare_tsne_data(data, max_samples=None):
    if data is None or len(data) == 0:
        return None

    if max_samples is None or len(data) <= max_samples:
        return data.reshape(data.shape[0], -1)

    indices = np.random.choice(len(data), max_samples, replace=False)
    sampled_data = data[indices]
    return sampled_data.reshape(sampled_data.shape[0], -1)


def load_generated_data(folder_path):
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


def plot_tsne(data_dict, building_name, sparsity):
    datasets = []
    data_types = []

    for data_type in ['Traindata', 'Testdata', 'DDPM', 'OURS']:
        if data_type in data_dict and data_dict[data_type] is not None:
            max_samples = 500 if data_type in ['DDPM', 'OURS'] else None
            prepared_data = prepare_tsne_data(data_dict[data_type], max_samples)
            if prepared_data is not None:
                datasets.append(prepared_data)
                data_types.append(data_type)

    if len(datasets) < 2:
        print(f"数据不足，无法为 {building_name}_{sparsity} 生成 t-SNE 图")
        return

    combined_data = np.vstack(datasets)
    scaled_data = StandardScaler().fit_transform(combined_data)

    tsne_results = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=1000,
        random_state=42
    ).fit_transform(scaled_data)

    labels = []
    for i, data_type in enumerate(data_types):
        labels.extend([data_type] * datasets[i].shape[0])

    tsne_df = pd.DataFrame({
        'TSNE-1': tsne_results[:, 0],
        'TSNE-2': tsne_results[:, 1],
        '数据类型': labels
    })

    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()

    sns.scatterplot(
        x='TSNE-1',
        y='TSNE-2',
        hue='数据类型',
        style='数据类型',
        data=tsne_df,
        palette=[COLOR_PALETTE[dt] for dt in tsne_df['数据类型'].unique()],
        s=120,
        alpha=0.8,
        ax=ax,
        markers={
            'Traindata': 'o',
            'Testdata': 's',
            'OURS': 'D',
            'DDPM': '^'
        }
    )

    plt.title(
        f't-SNE 分布可视化',
        fontsize=24,
        pad=15,
        weight='bold'
    )
    plt.xlabel('t-SNE 维度 1', fontsize=22, weight='bold')
    plt.ylabel('t-SNE 维度 2', fontsize=18, weight='bold')

    ax.legend(
        title='数据来源',
        title_fontsize=18,
        fontsize=18,
        loc='best',
        frameon=True,
        framealpha=0.9,
        edgecolor='black',
        markerscale=2.0
    )

    filename = f"{building_name.replace(' ', '_').replace('/', '_')}_sparsity_{sparsity}_tsne.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=600)
    print(f"已保存 t-SNE 图: {output_path}")
    plt.close()


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

        oridata_file = os.path.join(
            test_data_folder,
            test_folder,
            'samples',
            'energy_norm_truth_24_train.npy'
        )
        test_file = os.path.join(
            test_data_folder,
            test_folder,
            'samples',
            'energy_norm_truth_24_test.npy'
        )

        if not os.path.exists(oridata_file) or not os.path.exists(test_file):
            print("原始数据或测试数据文件不存在，跳过")
            continue

        oridata = np.load(oridata_file)
        test_data = np.load(test_file)

        ddpm_folder = os.path.join(base_dir, 'diffts', str(sparsity), building_name)
        ddpm_data = load_generated_data(ddpm_folder)
        print(f"加载 DDPM 样本数: {len(ddpm_data) if ddpm_data is not None else 0}")

        ours_folder = os.path.join(base_dir, 'diffts-fft', str(sparsity), building_name)
        ours_data = load_generated_data(ours_folder)
        print(f"加载 OURS 样本数: {len(ours_data) if ours_data is not None else 0}")

        data_dict = {
            'Traindata': oridata,
            'Testdata': test_data,
            'DDPM': ddpm_data,
            'OURS': ours_data
        }

        plot_tsne(data_dict, building_name, sparsity)

print("所有 t-SNE 图生成完成")
