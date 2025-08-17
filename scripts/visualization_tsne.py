import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl

# ==================== 全局样式设置 ====================
# 设置学术论文风格的绘图参数
plt.rcParams.update({
    'font.family': 'Times New Roman',  # 主字体
    'font.size': 10,  # 基础字体大小
    'axes.titlesize': 12,  # 标题大小
    'axes.labelsize': 11,  # 坐标轴标签大小
    'xtick.labelsize': 10,  # X轴刻度大小
    'ytick.labelsize': 10,  # Y轴刻度大小
    'legend.fontsize': 10,  # 图例大小
    'figure.dpi': 300,  # 输出分辨率
    'savefig.dpi': 300,
    'mathtext.fontset': 'stix',  # 数学公式字体
    'axes.grid': False,  # 关闭网格线（t-SNE不需要）
    'legend.frameon': True,  # 图例边框
    'legend.framealpha': 0.8,  # 图例透明度
    'legend.loc': 'best',  # 图例位置
    'axes.linewidth': 0.8,  # 坐标轴线宽
    'axes.edgecolor': 'black'  # 坐标轴颜色
})

# 创建专业配色方案 (色盲友好)
COLOR_PALETTE = {
    'oridata': '#1f77b4',  # 蓝色 - 原始数据
    'ours': '#ff7f0e',  # 橙色 - Ours方法生成的数据
    'timegan': '#2ca02c',  # 绿色 - TimeGAN方法生成的数据
    'testdata': '#d62728'  # 红色 - 测试数据
}

# 定义稀疏率和数据长度
sparsity_rates = [90, 70, 50, 30]
lengths = [720]
base_dir = '../../fakedata'
test_data_folder = '../../train_and_testdata'
output_dir = '../../results/tsne/timegan_and_ours'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)


# 数据处理函数 - 根据新要求修改采样策略
def prepare_tsne_data(data, max_samples=None):
    """准备用于t-SNE分析的数据"""
    if data is None or len(data) == 0:
        return None

    # 如果没有指定最大样本数或数据量小于最大样本数，返回所有数据
    if max_samples is None or len(data) <= max_samples:
        return data.reshape(data.shape[0], -1)

    # 否则随机采样
    indices = np.random.choice(len(data), max_samples, replace=False)
    sampled_data = data[indices]
    return sampled_data.reshape(sampled_data.shape[0], -1)


# t-SNE可视化函数 - 修改采样策略
def plot_tsne(data_dict, building_name, sparsity):
    """绘制高质量的t-SNE分布图"""
    # 首先确定timegan的数据量作为基准
    timegan_data = data_dict.get('timegan', None)
    max_samples = len(timegan_data) if timegan_data is not None and len(timegan_data) > 0 else None

    # 提取有效数据 - 使用timegan的数据量作为基准
    data_types = []
    datasets = []

    # 对于oridata和testdata，使用全部数据
    for data_type in ['oridata', 'testdata']:
        if data_type in data_dict and data_dict[data_type] is not None and len(data_dict[data_type]) > 0:
            prepared_data = prepare_tsne_data(data_dict[data_type], max_samples=None)
            if prepared_data is not None:
                data_types.append(data_type)
                datasets.append(prepared_data)

    # 对于ours和timegan，使用与timegan相同的数据量
    for data_type in ['ours', 'timegan']:
        if data_type in data_dict and data_dict[data_type] is not None and len(data_dict[data_type]) > 0:
            prepared_data = prepare_tsne_data(data_dict[data_type], max_samples=1000)
            if prepared_data is not None:
                data_types.append(data_type)
                datasets.append(prepared_data)

    if len(datasets) < 2:  # 至少需要两组数据
        print(f"数据不足，无法为{building_name}_{sparsity}生成t-SNE图")
        return

    # 合并数据
    combined_data = np.vstack(datasets)

    # 标准化数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)

    # 运行t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(scaled_data)

    # 创建数据标签
    labels = []
    start_idx = 0
    for i, data_type in enumerate(data_types):
        n_points = datasets[i].shape[0]
        labels.extend([data_type] * n_points)

    # 创建绘图数据框
    tsne_df = pd.DataFrame({
        'TSNE-1': tsne_results[:, 0],
        'TSNE-2': tsne_results[:, 1],
        'Data Type': labels
    })

    # 创建t-SNE图
    plt.figure(figsize=(10, 8), tight_layout=True)
    ax = plt.gca()

    # 使用自定义色板和样式
    sns.scatterplot(
        x='TSNE-1',
        y='TSNE-2',
        hue='Data Type',
        style='Data Type',
        data=tsne_df,
        palette=[COLOR_PALETTE[dt] for dt in tsne_df['Data Type'].unique()],
        s=80,  # 点的大小
        alpha=0.8,  # 透明度
        ax=ax,
        markers={'oridata': 'o', 'testdata': 's', 'ours': '^', 'timegan': 'D'}
    )

    # 设置标题和标签
    plt.title(f't-SNE Distribution - Building: {building_name} ({sparsity}% Sparsity)',
              fontsize=14, pad=15, weight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12, weight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=12, weight='bold')

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

    # 美化图形
    sns.despine(offset=10, trim=True)

    # 保存结果
    filename = f"{building_name.replace(' ', '_').replace('/', '_')}_sparsity_{sparsity}_tsne.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"已保存t-SNE图: {output_path}")
    plt.close()


# 主逻辑 - 遍历所有建筑和稀疏率
processed_buildings = set()

for test_folder in os.listdir(test_data_folder):
    parts = test_folder.split('_')
    if len(parts) >= 4:
        # 从文件夹名解析建筑名、长度和稀疏率
        building_name = '_'.join(parts[:-1])  # 保留建筑名部分
        length = int(parts[-2])
        sparsity = int(parts[-1])

        # 防止重复处理同一建筑
        key = f"{building_name}_{sparsity}"
        if key in processed_buildings:
            continue
        processed_buildings.add(key)

        if length == 720 and sparsity in sparsity_rates:
            print(f"处理建筑: {building_name}, 稀疏率: {sparsity}%")

            # 读取测试集数据
            test_file = os.path.join(test_data_folder, test_folder, 'samples', 'energy_norm_truth_24_test.npy')
            if not os.path.exists(test_file):
                print(f"测试文件 {test_file} 不存在, 跳过...")
                continue
            test_data = np.load(test_file)

            # 读取原始训练数据
            oridata_file = os.path.join(test_data_folder, test_folder, 'samples', 'energy_norm_truth_24_train.npy')
            if not os.path.exists(oridata_file):
                print(f"原始数据文件 {oridata_file} 不存在, 跳过...")
                continue
            oridata = np.load(oridata_file)

            # 读取Ours生成的合成数据
            ours_data = []
            sparsity_folder = os.path.join(base_dir, 'ours', str(sparsity), building_name)
            if os.path.exists(sparsity_folder):
                for sub_folder in os.listdir(sparsity_folder):
                    sub_folder_path = os.path.join(sparsity_folder, sub_folder)
                    if os.path.isdir(sub_folder_path):
                        for file in os.listdir(sub_folder_path):
                            if file.endswith('.npy'):
                                file_path = os.path.join(sub_folder_path, file)
                                if os.path.exists(file_path):
                                    try:
                                        data = np.load(file_path)
                                        ours_data.append(data)
                                    except Exception as e:
                                        print(f"加载Ours文件失败: {file_path}, 错误: {str(e)}")
                # 如果找到了ours数据，合并所有文件
                if ours_data:
                    ours_data = np.concatenate(ours_data, axis=0)
                    print(f"加载了 {len(ours_data)} 个Ours生成的样本")
                else:
                    ours_data = None
                    print(f"未找到有效的Ours数据")
            else:
                ours_data = None
                print(f"Ours文件夹不存在: {sparsity_folder}")

            # 读取TimeGAN生成的合成数据
            timegan_file = os.path.join(base_dir, 'timegan', str(sparsity), 'train', f'generated_{building_name}.npy')
            if os.path.exists(timegan_file):
                try:
                    timegan_data = np.load(timegan_file)
                    print(f"加载了 {len(timegan_data)} 个TimeGAN生成的样本")
                except Exception as e:
                    print(f"加载TimeGAN文件失败: {timegan_file}, 错误: {str(e)}")
                    timegan_data = None
            else:
                timegan_data = None
                print(f"TimeGAN文件不存在: {timegan_file}")
            diffts_data = []
            sparsity_folder = os.path.join(base_dir, 'diffts', str(sparsity), building_name)
            if os.path.exists(sparsity_folder):
                for sub_folder in os.listdir(sparsity_folder):
                    sub_folder_path = os.path.join(sparsity_folder, sub_folder)
                    if os.path.isdir(sub_folder_path):
                        for file in os.listdir(sub_folder_path):
                            if file.endswith('.npy'):
                                file_path = os.path.join(sub_folder_path, file)
                                if os.path.exists(file_path):
                                    try:
                                        data = np.load(file_path)
                                        diffts_data.append(data)
                                    except Exception as e:
                                        print(f"加载Ours文件失败: {file_path}, 错误: {str(e)}")
                # 如果找到了ours数据，合并所有文件
                if diffts_data:
                    diffts_data = np.concatenate(diffts_data, axis=0)
                    print(f"加载了 {len(ours_data)} 个Ours生成的样本")
                else:
                    diffts_data = None
                    print(f"未找到有效的Ours数据")
            else:
                diffts_data = None
                print(f"Ours文件夹不存在: {sparsity_folder}")
            # 创建数据字典
            data_dict = {
                'oridata': oridata,
                'testdata': test_data,
                'ours': ours_data,
                'timegan': diffts_data
            }

            # 生成并保存t-SNE图
            plot_tsne(data_dict, building_name, sparsity)

print("所有t-SNE图生成完成！")