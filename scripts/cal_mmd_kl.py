import os
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics.pairwise import rbf_kernel
import warnings

warnings.filterwarnings('ignore')

# 定义稀疏率和数据长度
sparsity_rates = [90, 70, 50, 30]
lengths = [4320]
methods = ['diffts-fft', 'diffts', 'timegan', 'cgan','wgan']
base_dir = '../fakedata'
test_data_folder = '../testdata'

# 计算 KL 散度
def compute_kl_divergence(data1, data2, bins=100):
    """
    计算两个数据集的 KL 散度，使用直方图估计概率分布。
    data1, data2: 形状为 (samples, time_steps, features) 的数组
    """
    # 展平数据为一维
    data1_flat = data1.reshape(-1)
    data2_flat = data2.reshape(-1)

    # 计算直方图
    hist1, bin_edges = np.histogram(data1_flat, bins=bins, density=True, range=(
        min(data1_flat.min(), data2_flat.min()), max(data1_flat.max(), data2_flat.max())))
    hist2, _ = np.histogram(data2_flat, bins=bin_edges, density=True)

    # 避免零概率，加小值平滑
    hist1 = hist1 + 1e-10
    hist2 = hist2 + 1e-10

    # 归一化
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()

    # 计算 KL 散度
    kl_div = entropy(hist1, hist2)
    return kl_div if np.isfinite(kl_div) else np.inf

# 计算 MMD（使用高斯核）
def compute_mmd(data1, data2, sigma=1.0):
    """
    计算两个数据集的 MMD，使用高斯核。
    data1, data2: 形状为 (samples, time_steps, features) 的数组
    """
    # 展平数据为 (samples, time_steps * features)
    data1_flat = data1.reshape(data1.shape[0], -1)
    data2_flat = data2.reshape(data2.shape[0], -1)

    # 取较小的样本数以避免内存问题
    n_samples = min(data1_flat.shape[0], data2_flat.shape[0])
    data1_flat = data1_flat[:n_samples]
    data2_flat = data2_flat[:n_samples]

    # 计算高斯核矩阵
    XX = rbf_kernel(data1_flat, data1_flat, gamma=1.0 / (2 * sigma ** 2))
    YY = rbf_kernel(data2_flat, data2_flat, gamma=1.0 / (2 * sigma ** 2))
    XY = rbf_kernel(data1_flat, data2_flat, gamma=1.0 / (2 * sigma ** 2))

    # 计算 MMD
    mmd = np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)
    return np.sqrt(max(0, mmd))  # 确保非负

# 主逻辑
results_all = []
for test_folder in os.listdir(test_data_folder):
    parts = test_folder.split('_')
    if len(parts) >= 4:
        building_name = '_'.join(parts[:-1])
        length = int(parts[-2])
        sparsity = int(parts[-1])

        if length == 4320 and sparsity in sparsity_rates:
            # 读取测试集数据
            test_file = os.path.join(test_data_folder, test_folder, 'samples', 'energy_norm_truth_24_test.npy')
            if not os.path.exists(test_file):
                print(f"Test file {test_file} not found, skipping...")
                continue
            test_data = np.load(test_file)
            if test_data.shape[1:] != (24, 6):
                print(f"Invalid shape for {test_file}: {test_data.shape}, skipping...")
                continue
            print(f"Loaded test data: {test_file}, shape: {test_data.shape}")

            # 读取 oridata（训练数据）
            oridata_file = os.path.join(test_data_folder, test_folder, 'samples', 'energy_norm_truth_24_train.npy')
            if not os.path.exists(oridata_file):
                print(f"Oridata file {oridata_file} not found, skipping...")
                continue
            oridata = np.load(oridata_file)
            if oridata.shape[1:] != (24, 6):
                print(f"Invalid shape for {oridata_file}: {oridata.shape}, skipping...")
                continue
            print(f"Loaded oridata: {oridata_file}, shape: {oridata.shape}")

            # 遍历每种合成数据方法
            for method in methods:
                print(f"Computing metrics for {building_name}, sparsity {sparsity}, method {method}...")
                metrics = {'Building': building_name, 'Sparsity': sparsity, 'Method': method}

                # 读取合成数据
                synth_data = None
                if method in ['diffts-fft', 'diffts']:
                    sparsity_folder = os.path.join(base_dir, method, str(sparsity), building_name)
                    for sub_folder in os.listdir(sparsity_folder):
                        sub_folder_path = os.path.join(sparsity_folder, sub_folder)
                        if os.path.isdir(sub_folder_path):
                            for file in os.listdir(sub_folder_path):
                                if file.endswith('.npy'):
                                    file_path = os.path.join(sub_folder_path, file)
                                    if os.path.exists(file_path):
                                        synth_data = np.load(file_path)
                                        if synth_data.shape[1:] != (24, 6):
                                            print(f"Invalid shape for {file_path}: {synth_data.shape}, skipping...")
                                            synth_data = None
                                        break
                elif method in ['timegan', 'cgan','wgan']:
                    sparsity_folder = os.path.join(base_dir, method, str(sparsity))
                    file_name = f'generated_{building_name}.npy'
                    file_path = os.path.join(sparsity_folder, 'train', file_name)
                    if os.path.exists(file_path):
                        synth_data = np.load(file_path)
                        if synth_data.shape[1:] != (24, 6):
                            print(f"Invalid shape for {file_path}: {synth_data.shape}, skipping...")
                            synth_data = None

                if synth_data is None:
                    print(f"Synthetic data for {method} not found or invalid, skipping...")
                    continue

                print(f"Loaded synthetic data: {file_path}, shape: {synth_data.shape}")

                # 计算与 oridata 的 KL 散度和 MMD
                metrics['KL_oridata'] = compute_kl_divergence(synth_data, oridata)
                metrics['MMD_oridata'] = compute_mmd(synth_data, oridata)

                # 计算与 test_data 的 KL 散度和 MMD
                metrics['KL_testdata'] = compute_kl_divergence(synth_data, test_data)
                metrics['MMD_testdata'] = compute_mmd(synth_data, test_data)

                print(f"Metrics for {building_name}, sparsity {sparsity}, method {method}: {metrics}")
                results_all.append(metrics)

# 保存所有结果到统一 CSV
if results_all:
    output_dir = '../results/distribution_4320'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results_df = pd.DataFrame(results_all)
    all_metrics_csv = os.path.join(output_dir, 'all_distribution_metrics.csv')
    results_df.to_csv(all_metrics_csv, index=False)
    print(f"Saved all results to {all_metrics_csv}")

    # 生成总结 CSV：按方法和稀疏率计算平均指标
    summary_df = results_df.groupby(['Method', 'Sparsity'])[['KL_oridata', 'MMD_oridata', 'KL_testdata', 'MMD_testdata']].mean().reset_index()
    summary_csv = os.path.join(output_dir, 'summary_metrics.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary metrics to {summary_csv}")