import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
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
    'Traindata': '#2ca02c',#2ca02c  # 蓝色 - 原始数据
    'Testdata': '#d62728',  # 红色 - 测试数据
    'CDDM': '#ff7f0e',  # 橙色 - DDPM方法
    'OURS': '#1f77b4'  # 绿色 - OURS方法
}

# 定义稀疏率和路径（保持不变）
sparsity_rates = [500,100,300]
base_dir = '../fakedata2'
test_data_folder = '../testdata2'
output_dir = '../results/tsne/260407_outline'
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


    for data_type in ['Traindata', 'Testdata', 'CDDM', 'OURS']:
        if data_type in data_dict and data_dict[data_type] is not None:
            # 修改点1：所有数据统一最多200
            prepared_data = prepare_tsne_data(data_dict[data_type], max_samples=200)
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
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()

    # 绘制散点图（点大小从120缩小到80）
    sns.scatterplot(
        x='TSNE-1',
        y='TSNE-2',
        hue='Data Type',
        style='Data Type',
        data=tsne_df,
        palette=[COLOR_PALETTE[dt] for dt in tsne_df['Data Type'].unique()],
        s=120,
        alpha=0.5,  # 空心点建议不透明
        ax=ax,
        markers={'Traindata': 'o', 'Testdata': 's', 'OURS': 'D', 'CDDM': '^'},legend=False,
        edgecolor='black',  # 边框颜色
        facecolors='none',  # 关键：空心
        linewidth=1.2  # 边框粗一点更清晰
    )
    # ==================== 画每一类的外包线（Convex Hull）====================
    for data_type in tsne_df['Data Type'].unique():
        subset = tsne_df[tsne_df['Data Type'] == data_type]

        if len(subset) < 3:
            continue  # 少于3个点无法构成凸包

        points = subset[['TSNE-1', 'TSNE-2']].values

        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]

            # 闭合多边形
            hull_points = np.vstack([hull_points, hull_points[0]])

            ax.plot(
                hull_points[:, 0],
                hull_points[:, 1],
                color=COLOR_PALETTE[data_type],
                linewidth=2.0,
                alpha=0.9
            )
        except:
            pass  # 防止极端情况报错
    # for data_type in tsne_df['Data Type'].unique():
    #     subset = tsne_df[tsne_df['Data Type'] == data_type]
    #
    #     ax.scatter(
    #         subset['TSNE-1'],
    #         subset['TSNE-2'],
    #         label=data_type,
    #         marker={'Traindata': 'o', 'Testdata': 's', 'OURS': 'D', 'CDDM': '^'}[data_type],
    #         s=120,
    #         facecolors='none',  # 空心
    #         edgecolors=COLOR_PALETTE[data_type],  # 用类别颜色作为边框
    #         linewidths=1.2,
    #         alpha=0.9
    #     )

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
        sparsity = int(parts[-1])

        key = f"{building_name}_{sparsity}"
        if key in processed_buildings or sparsity not in sparsity_rates:
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
        ddpm_folder = os.path.join(base_dir, 'CDDM', str(sparsity), building_name)
        ddpm_data = load_generated_data(ddpm_folder)
        print(f"加载了 {len(ddpm_data) if ddpm_data is not None else 0} 个DDPM生成的样本")

        ours_folder = os.path.join(base_dir, 'ours', str(sparsity), building_name)
        ours_data = load_generated_data(ours_folder)
        print(f"加载了 {len(ours_data) if ours_data is not None else 0} 个OURS生成的样本")

        # 生成t-SNE图
        data_dict = {
            'Traindata': oridata,
            'Testdata': test_data,
            'CDDM': ddpm_data,
            'OURS': ours_data
        }
        plot_tsne(data_dict, building_name, sparsity)

print("所有t-SNE图生成完成！")