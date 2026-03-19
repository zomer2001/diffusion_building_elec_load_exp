import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 17,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'figure.dpi': 600,
    'savefig.dpi': 600
})

COLOR_PALETTE = {
    'Traindata': '#1f77b4',
    'Testdata': '#d62728',
    'DDPM': '#ff7f0e',
    'OURS': '#2ca02c'
}

sparsity_rates = [70]
base_dir = '../fakedata'
test_data_folder = '../testdata'
output_dir = '../results/tsne/compare_three_lunwen1'
os.makedirs(output_dir, exist_ok=True)


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
                for f in os.listdir(sub_path):
                    if f.endswith('.npy'):
                        try:
                            data.append(np.load(os.path.join(sub_path, f)))
                        except:
                            pass
    return np.concatenate(data, axis=0) if data else None


# ==================== 核心：一次计算 t-SNE ====================
def compute_tsne_all(data_dict):
    datasets = []
    labels = []

    for key in ['Traindata', 'Testdata', 'DDPM', 'OURS']:
        if key in data_dict and data_dict[key] is not None:
            max_samples = 500 if key in ['DDPM', 'OURS'] else None
            d = prepare_tsne_data(data_dict[key], max_samples)
            if d is not None:
                datasets.append(d)
                labels += [key] * len(d)

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


# ==================== 画子图 ====================
def plot_subset(tsne_df, selected_types, title, save_path):
    df = tsne_df[tsne_df['Data Type'].isin(selected_types)]

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    sns.scatterplot(
        data=df,
        x='TSNE-1',
        y='TSNE-2',
        hue='Data Type',
        style='Data Type',
        palette=[COLOR_PALETTE[t] for t in selected_types],
        s=80,
        alpha=0.8,
        ax=ax,
        markers={'Traindata': 'o', 'Testdata': 's', 'OURS': 'D', 'DDPM': '^'}
    )

    plt.title(title, fontsize=20, weight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    plt.legend(frameon=True, edgecolor='black')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


# ==================== 主流程 ====================
processed = set()

for test_folder in os.listdir(test_data_folder):
    parts = test_folder.split('_')
    if len(parts) < 4:
        continue

    building_name = '_'.join(parts[:-1])
    length = int(parts[-2])
    sparsity = int(parts[-1])

    key = f"{building_name}_{sparsity}"
    if key in processed or length != 2160 or sparsity not in sparsity_rates:
        continue

    processed.add(key)

    print(f"处理: {building_name}")

    train_file = os.path.join(test_data_folder, test_folder, 'samples', 'energy_norm_truth_24_train.npy')
    test_file = os.path.join(test_data_folder, test_folder, 'samples', 'energy_norm_truth_24_test.npy')

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        continue

    train_data = np.load(train_file)
    test_data = np.load(test_file)

    ddpm_data = load_generated_data(os.path.join(base_dir, 'diffts', str(sparsity), building_name))
    ours_data = load_generated_data(os.path.join(base_dir, 'diffts-fft', str(sparsity), building_name))

    data_dict = {
        'Traindata': train_data,
        'Testdata': test_data,
        'DDPM': ddpm_data,
        'OURS': ours_data
    }

    # ⭐ 一次性计算
    tsne_df = compute_tsne_all(data_dict)

    # ⭐ 三张图
    base_name = building_name.replace(' ', '_')

    plot_subset(
        tsne_df,
        ['Traindata', 'Testdata'],
        f'{building_name} (Real vs Test)',
        os.path.join(output_dir, f'{base_name}_real_test.png')
    )

    plot_subset(
        tsne_df,
        ['Traindata', 'Testdata', 'DDPM'],
        f'{building_name} (+DDPM)',
        os.path.join(output_dir, f'{base_name}_ddpm.png')
    )

    plot_subset(
        tsne_df,
        ['Traindata', 'Testdata', 'OURS'],
        f'{building_name} (+OURS)',
        os.path.join(output_dir, f'{base_name}_ours.png')
    )

print("完成")