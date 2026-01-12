import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ==================== 全局样式设置（字体增大）====================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 16,  # 基础字体从14增大到16
    'axes.titlesize': 18,  # 轴标题从16增大到18
    'axes.labelsize': 17,  # 轴标签从15增大到17
    'xtick.labelsize': 15,  # 刻度标签从13增大到15
    'ytick.labelsize': 15,
    'legend.fontsize': 15,  # 图例字体从13增大到15
    'figure.dpi': 600,  # 保持高分辨率
    'savefig.dpi': 600,
    'mathtext.fontset': 'stix',
    'axes.grid': False,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.loc': 'best',
    'axes.linewidth': 1.0,  # 轴线宽度不变
    'axes.edgecolor': 'black'
})

# 配色方案（保持不变）
COLOR_PALETTE = {
    'oridata': '#1f77b4',  # 蓝色 - 原始数据
    'testdata': '#d62728',  # 红色 - 测试数据
    'DDPM': '#ff7f0e',  # 橙色 - DDPM方法
    'OURS': '#2ca02c'  # 绿色 - OURS方法
}

# 定义稀疏率和路径（保持不变）
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
    """绘制t-SNE分布图（字体增大+点缩小+正方形比例）"""
    # 准备数据（保持不变）
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

    # 合并和标准化数据（保持不变）
    combined_data = np.vstack(datasets)
    scaled_data = StandardScaler().fit_transform(combined_data)

    # 运行t-SNE（保持不变）
    tsne_results = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42).fit_transform(scaled_data)

    # 创建绘图数据框（保持不变）
    labels = []
    for i, data_type in enumerate(data_types):
        labels.extend([data_type] * datasets[i].shape[0])

    tsne_df = pd.DataFrame({
        'TSNE-1': tsne_results[:, 0],
        'TSNE-2': tsne_results[:, 1],
        'Data Type': labels
    })

    # 修改：将图片尺寸改为正方形 (10, 10) 保持1:1比例
    plt.figure(figsize=(10, 8), tight_layout=True)
    ax = plt.gca()

    # 绘制散点图（点大小从120缩小到80）
    sns.scatterplot(
        x='TSNE-1',
        y='TSNE-2',
        hue='Data Type',
        style='Data Type',
        data=tsne_df,
        palette=[COLOR_PALETTE[dt] for dt in tsne_df['Data Type'].unique()],
        s=80,  # 点大小缩小（原120→80）
        alpha=0.8,  # 透明度不变，确保点重叠时仍可区分
        ax=ax,
        markers={'oridata': 'o', 'testdata': 's', 'OURS': 'D', 'DDPM': '^'}
    )

    # 标题字体进一步增大到20号（原18→20）
    plt.title(f't-SNE Distribution:{building_name}',
              fontsize=24, pad=15, weight='bold')
    # 轴标签字体增大（原14→17，与全局设置一致）
    plt.xlabel('t-SNE Dimension 1', fontsize=22, weight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=18, weight='bold')

    # 图例字体增大（与全局设置一致）
    legend = ax.legend(
        title='Data Type',
        title_fontsize=18,  # 图例标题从14增大到16
        fontsize=18,  # 图例内容从13增大到15
        loc='best',
        frameon=True,
        framealpha=0.9,
        edgecolor='black',
        markerscale=2.0  # 图例标记比例不变，确保清晰
    )

    # 保存图像（保持不变）
    filename = f"{building_name.replace(' ', '_').replace('/', '_')}_sparsity_{sparsity}_tsne.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=600)
    print(f"已保存t-SNE图: {output_path}")
    plt.close()


# 主逻辑：遍历数据并生成t-SNE图（保持不变）
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
        ddpm_folder = os.path.join(base_dir, 'diffts-fft', str(sparsity), building_name)
        ddpm_data = load_generated_data(ddpm_folder)
        print(f"加载了 {len(ddpm_data) if ddpm_data is not None else 0} 个DDPM生成的样本")

        ours_folder = os.path.join(base_dir, 'ours_gen', str(sparsity), building_name)
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