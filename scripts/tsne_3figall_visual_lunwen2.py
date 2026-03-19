import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ==================== 全局样式 ====================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 17,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'mathtext.fontset': 'stix',
    'axes.grid': False,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'axes.linewidth': 1.0,
    'axes.edgecolor': 'black'
})

COLOR_PALETTE = {
    'Traindata': '#1f77b4',
    'Testdata': '#d62728',
    'CDDM': '#ff7f0e',
    'OURS': '#2ca02c'
}

sparsity_rates = [70]
base_dir = '../fakedata'
test_data_folder = '../testdata'
output_dir = '../results/tsne/three_compare_lunwen2'
os.makedirs(output_dir, exist_ok=True)


# ==================== 数据处理 ====================
def prepare_tsne_data(data, max_samples=None):
    if data is None or len(data) == 0:
        return None

    if max_samples and len(data) > max_samples:
        idx = np.random.choice(len(data), max_samples, replace=False)
        data = data[idx]

    return data.reshape(data.shape[0], -1)


def load_generated_data(folder_path):
    data = []
    if os.path.exists(folder_path):
        for sub_folder in os.listdir(folder_path):
            sub_path = os.path.join(folder_path, sub_folder)
            if os.path.isdir(sub_path):
                for file in os.listdir(sub_path):
                    if file.endswith('.npy'):
                        try:
                            data.append(np.load(os.path.join(sub_path, file)))
                        except:
                            pass
    return np.concatenate(data, axis=0) if data else None


# ==================== 核心：一次t-SNE ====================
def compute_tsne_all(data_dict):
    datasets = []
    labels = []

    for key in ['Traindata', 'Testdata', 'CDDM', 'OURS']:
        if data_dict.get(key) is not None:
            max_samples = 500 if key in ['CDDM', 'OURS'] else None
            d = prepare_tsne_data(data_dict[key], max_samples)
            if d is not None:
                datasets.append(d)
                labels += [key] * len(d)

    if len(datasets) < 2:
        return None

    combined = np.vstack(datasets)
    scaled = StandardScaler().fit_transform(combined)

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    result = tsne.fit_transform(scaled)

    df = pd.DataFrame({
        'TSNE-1': result[:, 0],
        'TSNE-2': result[:, 1],
        'Data Type': labels
    })

    return df


# ==================== 子图绘制 ====================
def plot_subset(tsne_df, selected_types, title, save_path):
    df = tsne_df[tsne_df['Data Type'].isin(selected_types)]

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    sns.scatterplot(
        x='TSNE-1',
        y='TSNE-2',
        hue='Data Type',
        style='Data Type',
        data=df,
        palette=[COLOR_PALETTE[t] for t in selected_types],
        s=120,
        alpha=0.8,
        ax=ax,
        markers={'Traindata': 'o', 'Testdata': 's', 'OURS': 'D', 'CDDM': '^'}
    )

    plt.title(title, fontsize=24, weight='bold', pad=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=22, weight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=18, weight='bold')

    ax.legend(
        title='Data Type',
        title_fontsize=18,
        fontsize=18,
        frameon=True,
        edgecolor='black'
    )

    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.close()


# ==================== 主流程 ====================
processed_buildings = set()

for test_folder in os.listdir(test_data_folder):

    parts = test_folder.split('_')
    if len(parts) < 4:
        continue

    building_name = '_'.join(parts[:-1])
    length = int(parts[-2])
    sparsity = int(parts[-1])

    key = f"{building_name}_{sparsity}"
    if key in processed_buildings or length != 2160 or sparsity not in sparsity_rates:
        continue

    processed_buildings.add(key)

    print(f"处理建筑: {building_name}")

    train_file = os.path.join(test_data_folder, test_folder, 'samples', 'energy_norm_truth_24_train.npy')
    test_file = os.path.join(test_data_folder, test_folder, 'samples', 'energy_norm_truth_24_test.npy')

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        continue

    train_data = np.load(train_file)
    test_data = np.load(test_file)

    cddm_data = load_generated_data(os.path.join(base_dir, 'diffts-fft', str(sparsity), building_name))
    ours_data = load_generated_data(os.path.join(base_dir, 'ours_gen', str(sparsity), building_name))

    print(f"CDDM: {0 if cddm_data is None else len(cddm_data)}")
    print(f"OURS: {0 if ours_data is None else len(ours_data)}")

    data_dict = {
        'Traindata': train_data,
        'Testdata': test_data,
        'CDDM': cddm_data,
        'OURS': ours_data
    }

    # ⭐ 一次计算
    tsne_df = compute_tsne_all(data_dict)
    if tsne_df is None:
        continue

    base = building_name.replace(' ', '_').replace('/', '_')

    # ⭐ 三张图
    plot_subset(
        tsne_df,
        ['Traindata', 'Testdata'],
        f'{building_name} (Real vs Test)',
        os.path.join(output_dir, f'{base}_real_test.png')
    )

    plot_subset(
        tsne_df,
        ['Traindata', 'Testdata', 'CDDM'],
        f'{building_name} (+CDDM)',
        os.path.join(output_dir, f'{base}_cddm.png')
    )

    plot_subset(
        tsne_df,
        ['Traindata', 'Testdata', 'OURS'],
        f'{building_name} (+OURS)',
        os.path.join(output_dir, f'{base}_ours.png')
    )

print("全部完成")