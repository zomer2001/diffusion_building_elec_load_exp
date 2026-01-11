import os
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics.pairwise import rbf_kernel
import warnings

warnings.filterwarnings('ignore')

# ================= 基本配置 =================
sparsity_rates = [90, 70, 50, 30]
lengths = [2160]
methods = ['ours']

base_dir = '../fakedata'
test_data_folder = '../testdata'


# ================= 指标计算函数 =================
def compute_kl_divergence(data1, data2, bins=100):
    """
    使用直方图估计概率分布计算 KL 散度
    data shape: (samples, time_steps, features)
    """
    data1_flat = data1.reshape(-1)
    data2_flat = data2.reshape(-1)

    hist1, bin_edges = np.histogram(
        data1_flat,
        bins=bins,
        density=True,
        range=(min(data1_flat.min(), data2_flat.min()),
               max(data1_flat.max(), data2_flat.max()))
    )
    hist2, _ = np.histogram(data2_flat, bins=bin_edges, density=True)

    hist1 += 1e-10
    hist2 += 1e-10

    hist1 /= hist1.sum()
    hist2 /= hist2.sum()

    kl_div = entropy(hist1, hist2)
    return kl_div if np.isfinite(kl_div) else np.inf


def compute_mmd(data1, data2, sigma=1.0):
    """
    使用高斯核计算 MMD
    """
    data1_flat = data1.reshape(data1.shape[0], -1)
    data2_flat = data2.reshape(data2.shape[0], -1)

    n_samples = min(data1_flat.shape[0], data2_flat.shape[0])
    data1_flat = data1_flat[:n_samples]
    data2_flat = data2_flat[:n_samples]

    gamma = 1.0 / (2 * sigma ** 2)
    XX = rbf_kernel(data1_flat, data1_flat, gamma=gamma)
    YY = rbf_kernel(data2_flat, data2_flat, gamma=gamma)
    XY = rbf_kernel(data1_flat, data2_flat, gamma=gamma)

    mmd = np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)
    return np.sqrt(max(0, mmd))


# ================= 主逻辑 =================
results_all = []

for test_folder in os.listdir(test_data_folder):
    parts = test_folder.split('_')
    if len(parts) < 4:
        continue

    building_name = '_'.join(parts[:-1])
    length = int(parts[-2])
    sparsity = int(parts[-1])

    if length != 2160 or sparsity not in sparsity_rates:
        continue

    # ---------- 读取 test data ----------
    test_file = os.path.join(
        test_data_folder, test_folder, 'samples',
        'energy_norm_truth_24_test.npy'
    )
    if not os.path.exists(test_file):
        print(f"[Skip] Test file not found: {test_file}")
        continue

    test_data = np.load(test_file)
    if test_data.shape[1:] != (24, 6):
        print(f"[Skip] Invalid test shape: {test_data.shape}")
        continue

    # ---------- 读取 oridata (train) ----------
    oridata_file = os.path.join(
        test_data_folder, test_folder, 'samples',
        'energy_norm_truth_24_train.npy'
    )
    if not os.path.exists(oridata_file):
        print(f"[Skip] Oridata file not found: {oridata_file}")
        continue

    oridata = np.load(oridata_file)
    if oridata.shape[1:] != (24, 6):
        print(f"[Skip] Invalid oridata shape: {oridata.shape}")
        continue

    print(f"\n[OK] Loaded test & train for {building_name}, sparsity {sparsity}")

    # ---------- 遍历方法 ----------
    for method in methods:
        metrics = {
            'Building': building_name,
            'Sparsity': sparsity,
            'Method': method
        }

        synth_data = None
        file_path = None

        # ========== ours / diffts 系列 ==========
        if method in ['diffts-fft', 'diffts', 'ours']:
            sparsity_folder = os.path.join(
                base_dir, method, str(sparsity), building_name
            )

            # ⭐ 关键：先判断目录是否存在
            if not os.path.exists(sparsity_folder):
                print(f"[Skip] No synthetic folder: {sparsity_folder}")
                continue

            for sub_folder in os.listdir(sparsity_folder):
                sub_folder_path = os.path.join(sparsity_folder, sub_folder)
                if not os.path.isdir(sub_folder_path):
                    continue

                for file in os.listdir(sub_folder_path):
                    if file.endswith('.npy'):
                        file_path = os.path.join(sub_folder_path, file)
                        synth_data = np.load(file_path)

                        if synth_data.shape[1:] != (24, 6):
                            print(f"[Skip] Invalid synth shape: {synth_data.shape}")
                            synth_data = None
                        break

                if synth_data is not None:
                    break

        # ========== GAN 系列（保留兼容） ==========
        elif method in ['timegan', 'cgan', 'wgan']:
            sparsity_folder = os.path.join(base_dir, method, str(sparsity))
            file_name = f'generated_{building_name}.npy'
            file_path = os.path.join(sparsity_folder, 'train', file_name)

            if os.path.exists(file_path):
                synth_data = np.load(file_path)
                if synth_data.shape[1:] != (24, 6):
                    print(f"[Skip] Invalid synth shape: {synth_data.shape}")
                    synth_data = None

        if synth_data is None:
            print(f"[Skip] No valid synthetic data for {building_name}, {method}")
            continue

        print(f"[OK] Loaded synthetic data: {file_path}, shape {synth_data.shape}")

        # ---------- 计算指标 ----------
        metrics['KL_oridata'] = compute_kl_divergence(synth_data, oridata)
        metrics['MMD_oridata'] = compute_mmd(synth_data, oridata)
        metrics['KL_testdata'] = compute_kl_divergence(synth_data, test_data)
        metrics['MMD_testdata'] = compute_mmd(synth_data, test_data)

        print(f"[Done] Metrics: {metrics}")
        results_all.append(metrics)


# ================= 保存结果 =================
if results_all:
    output_dir = '../results/distribution_2160_2_newours_0111'
    os.makedirs(output_dir, exist_ok=True)

    results_df = pd.DataFrame(results_all)
    all_csv = os.path.join(output_dir, 'all_distribution_metrics.csv')
    results_df.to_csv(all_csv, index=False)
    print(f"\n[Saved] {all_csv}")

    summary_df = (
        results_df
        .groupby(['Method', 'Sparsity'])[
            ['KL_oridata', 'MMD_oridata', 'KL_testdata', 'MMD_testdata']
        ]
        .mean()
        .reset_index()
    )
    summary_csv = os.path.join(output_dir, 'summary_metrics.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"[Saved] {summary_csv}")
