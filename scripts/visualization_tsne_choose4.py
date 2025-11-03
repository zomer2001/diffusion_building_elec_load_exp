import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ==================== 全局样式设置 ====================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 14,  # 基础字体大小
    'axes.titlesize': 16,  # 轴标题字体
    'axes.labelsize': 15,  # 轴标签字体
    'xtick.labelsize': 13,  # 刻度标签字体
    'ytick.labelsize': 13,
    'legend.fontsize': 13,  # 图例字体
    'figure.dpi': 1200,  # 高分辨率
    'savefig.dpi': 1200,
    'mathtext.fontset': 'stix',
    'axes.grid': False,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.loc': 'best',
    'axes.linewidth': 1.0,  # 轴线宽度
    'axes.edgecolor': 'black'
})

# 配色方案
COLOR_PALETTE = {
    'oridata': '#1f77b4',  # 蓝色 - 原始数据
    'testdata': '#d62728',  # 红色 - 测试数据
    'DDPM': '#ff7f0e',  # 橙色 - DDPM方法
    'OURS': '#2ca02c'  # 绿色 - OURS方法
}

# 定义稀疏率和路径
sparsity_rates = [70]
base_dir = '../fakedata'
test_data_folder = '../testdata'
output_dir = '../results/tsne/ddpm_and_ours'
os.makedirs(output_dir, exist_ok=True)


def prepare_tsne_data(data, max_samples=None):
    """准备t-SNE输入数据（展平+采样）"""
    if data is None or len(data) == 0:
        return None

    if max_samples is None or len(data) <= max_samples:
        return data.reshape(data.shape[0], -1)

    indices = np.random.choice(len(data), max_samples, replace=False)
    sampled_data = data[indices]
    return sampled_data.reshape(sampled_data.shape[0], -1)


def load_generated_data(folder_path):
    """加载生成的合成数据（.npy文件）"""
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
    """绘制t-SNE分布图（调整为长方形+大标题）"""
    # 准备数据
    datasets = []
    data_types = []

    for data_type in ['oridata', 'testdata', 'DDPM', 'OURS']:
        if data_type in data_dict and data_dict[data_type] is not None:
            max_samples = 500 if data_type in ['DDPM', 'OURS'] else None
            prepared_data = prepare_tsne_data(data_dict[data_type], max_samples)
            if prepared_data is not None:
                datasets.append(prepared_data)
                data_types.append(data_type)

    if len(datasets) < 2:
        print(f"数据不足，无法为{building_name}_{sparsity}生成t-SNE图")
        return

    # 合并和标准化数据
    combined_data = np.vstack(datasets)
    scaled_data = StandardScaler().fit_transform(combined_data)

    # 运行t-SNE
    tsne_results = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42).fit_transform(scaled_data)

    # 创建绘图数据框
    labels = []
    for i, data_type in enumerate(data_types):
        labels.extend([data_type] * datasets[i].shape[0])

    tsne_df = pd.DataFrame({
        'TSNE-1': tsne_results[:, 0],
        'TSNE-2': tsne_results[:, 1],
        'Data Type': labels
    })

    # 绘制t-SNE图（改为长方形：宽12，高6）
    plt.figure(figsize=(12, 6), tight_layout=True)  # 长方形尺寸（宽>高）
    ax = plt.gca()

    # 绘制散点图
    sns.scatterplot(
        x='TSNE-1',
        y='TSNE-2',
        hue='Data Type',
        style='Data Type',
        data=tsne_df,
        palette=[COLOR_PALETTE[dt] for dt in tsne_df['Data Type'].unique()],
        s=120,  # 点大小
        alpha=0.8,
        ax=ax,
        markers={'oridata': 'o', 'testdata': 's', 'OURS': 'D', 'DDPM': '^'}
    )

    # 标题字体增大到18号
    plt.title(f't-SNE Distribution - Building: {building_name} ({sparsity}% Sparsity)',
              fontsize=18, pad=15, weight='bold')  # 标题字体18号
    plt.xlabel('t-SNE Dimension 1', fontsize=14, weight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=14, weight='bold')

    # 图例设置
    legend = ax.legend(
        title='Data Type',
        title_fontsize=14,
        fontsize=13,
        loc='best',
        frameon=True,
        framealpha=0.9,
        edgecolor='black',
        markerscale=2.0
    )

    # 保存图像
    filename = f"{building_name.replace(' ', '_').replace('/', '_')}_sparsity_{sparsity}_tsne.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=1200)
    print(f"已保存t-SNE图: {output_path}")
    plt.close()


# 主逻辑：遍历数据并生成t-SNE图
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

        # 加载DDPM和OURS生成数据
        ddpm_folder = os.path.join(base_dir, 'diffts', str(sparsity), building_name)
        ddpm_data = load_generated_data(ddpm_folder)
        print(f"加载了 {len(ddpm_data) if ddpm_data is not None else 0} 个DDPM生成的样本")

        ours_folder = os.path.join(base_dir, 'diffts-fft', str(sparsity), building_name)
        ours_data = load_generated_data(ours_folder)
        print(f"加载了 {len(ours_data) if ours_data is not None else 0} 个OURS生成的样本")

        # 生成t-SNE图
        data_dict = {
            'oridata': oridata,
            'testdata': test_data,
            'DDPM': ddpm_data,
            'OURS': ours_data
        }
        plot_tsne(data_dict, building_name, sparsity)

print("所有t-SNE图生成完成！")