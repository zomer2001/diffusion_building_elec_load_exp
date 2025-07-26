import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置后端
matplotlib.use('TkAgg')  # 或者 'MacOSX'

# 路径设置
oridata_path = '/Users/zomeryang/Documents/bearlab/keti/project/exp/24_real_exp/train_and_testdata/Cockatoo_industrial_Nathaniel_start8000_2160_50/samples/energy_norm_truth_24_train.npy'
testdata_path = '/Users/zomeryang/Documents/bearlab/keti/project/exp/24_real_exp/train_and_testdata/Cockatoo_industrial_Nathaniel_start8000_2160_50/samples/energy_norm_truth_24_test.npy'
ours_path = '/Users/zomeryang/Documents/bearlab/keti/project/exp/24_real_exp/fakedata/ours/50/Cockatoo_industrial_Nathaniel_start8000_2160/Cockatoo_industrial_Nathaniel_start8000_2160_75/ddpm_fake_Cockatoo_industrial_Nathaniel_start8000_2160_75.npy'
diffts_path = '/Users/zomeryang/Documents/bearlab/keti/project/exp/24_real_exp/fakedata/diffts/50/Cockatoo_industrial_Nathaniel_start8000_2160/Cockatoo_industrial_Nathaniel_start8000_2160_50/ddpm_fake_Cockatoo_industrial_Nathaniel_start8000_2160_50.npy'


def load_data(path):
    """读取npy文件"""
    return np.load(path)


def find_most_similar_load(target_load, candidates_load, num_samples=5):
    """寻找最相似的负荷数据"""
    diff = np.abs(candidates_load - target_load)  # 计算差值
    total_diff = np.sum(diff, axis=1)  # 计算每行的差值绝对和
    min_indices = np.argsort(total_diff)[:num_samples]  # 取差值最小的前几个
    return candidates_load[min_indices], min_indices  # 返回相似样本及其索引
def find_most_similar_load2(target_load, candidates_load, num_samples=5):
    """寻找最相似的负荷数据（支持多个目标负荷）"""
    # 计算每个目标负荷与候选负荷之间的差异
    diffs = np.abs(candidates_load[:, np.newaxis] - target_load)  # 扩展维度进行广播计算
    total_diffs = np.sum(diffs, axis=2)  # 按列求和，得到每个候选负荷与所有目标负荷的总差异

    # 对每个候选负荷，计算其与所有目标负荷的差异和，选出最相似的负荷
    min_indices = np.argsort(np.sum(total_diffs, axis=1))[:num_samples]  # 取差值最小的前几个候选负荷的索引
    return candidates_load[min_indices], min_indices  # 返回相似样本及其索引

def plot_comparison(oridata_sample, ours_similar, diffts_similar, testdata_similar, sample_index):
    """绘制原始负荷数据与最相似负荷数据的对比图"""
    fig = plt.figure(figsize=(16, 12))  # 增加高度以适应更多行

    # 第一行：原始数据与生成数据对比 (左右并排)
    plt.subplot(4, 2, 1)  # 第一行左图
    for idx, similar_sample in enumerate(ours_similar):
        plt.plot(similar_sample, label=f'Ours Similar {idx + 1}')
    plt.plot(oridata_sample, label='Original', linestyle='--', color='black', linewidth=2)
    plt.title(f'Ours Comparison - Sample {sample_index + 1}')
    plt.ylabel('Load')
    #plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 2)  # 第一行右图
    for idx, similar_sample in enumerate(diffts_similar):
        plt.plot(similar_sample, label=f'diffTS Similar {idx + 1}')
    plt.plot(oridata_sample, label='Original', linestyle='--', color='black', linewidth=2)
    plt.title(f'diffTS Comparison - Sample {sample_index + 1}')
    #plt.legend()
    plt.grid(True)

    # 第二行：testdata与生成数据对比 (左右并排)
    plt.subplot(4, 2, 3)  # 第二行左图
    for idx, similar_sample in enumerate(ours_similar):
        plt.plot(similar_sample, label=f'Ours Similar {idx + 1}')
    plt.plot(testdata_similar[0], label='Test Data', linestyle='--', color='black', linewidth=2)
    plt.title(f'Ours vs Test Data')
    plt.ylabel('Load')
    #plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 4)  # 第二行右图
    for idx, similar_sample in enumerate(diffts_similar):
        plt.plot(similar_sample, label=f'diffTS Similar {idx + 1}')
    plt.plot(testdata_similar[0], label='Test Data', linestyle='--', color='black', linewidth=2)
    plt.title(f'diffTS vs Test Data')
    #plt.legend()
    plt.grid(True)

    # 新增第三行：testdata最相似的样本对比 (左右并排)
    plt.subplot(4, 2, 5)  # 第三行左图
    for idx, similar_sample in enumerate(ours_similar):
        plt.plot(similar_sample, label=f'Ours Similar {idx + 1}')
    plt.plot(testdata_similar[1], label='Closest Test Data 1', linestyle='--', color='green', linewidth=2)
    plt.title(f'Ours vs Closest Test Data 1')
    plt.ylabel('Load')
    #plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 6)  # 第三行右图
    for idx, similar_sample in enumerate(diffts_similar):
        plt.plot(similar_sample, label=f'diffTS Similar {idx + 1}')
    plt.plot(testdata_similar[1], label='Closest Test Data 1', linestyle='--', color='green', linewidth=2)
    plt.title(f'diffTS vs Closest Test Data 1')
    #plt.legend()
    plt.grid(True)

    # 新增第四行：testdata第二相似的样本对比 (左右并排)
    plt.subplot(4, 2, 7)  # 第四行左图
    for idx, similar_sample in enumerate(ours_similar):
        plt.plot(similar_sample, label=f'Ours Similar {idx + 1}')
    plt.plot(testdata_similar[2], label='Closest Test Data 2', linestyle='--', color='purple', linewidth=2)
    plt.title(f'Ours vs Closest Test Data 2')
    plt.xlabel('Time')
    plt.ylabel('Load')
    #plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 8)  # 第四行右图
    for idx, similar_sample in enumerate(diffts_similar):
        plt.plot(similar_sample, label=f'diffTS Similar {idx + 1}')
    plt.plot(testdata_similar[2], label='Closest Test Data 2', linestyle='--', color='purple', linewidth=2)
    plt.title(f'diffTS vs Closest Test Data 2')
    plt.xlabel('Time')
    #plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def filter_varying_samples(data_load):
    """
    筛选具有变化的样本（非恒定值序列）

    条件：
    1. 不是全零样本
    2. 不是所有值都相同的恒定值序列
    3. 序列中存在前后不同的值（变化）
    """
    varying_indices = []
    for idx, sample in enumerate(data_load):
        # 检查样本是否为全零或恒定值
        if np.all(sample == 0) or np.all(sample == sample[0]):
            continue

        # 检查序列中是否有变化（相邻值不同）
        has_variation = False
        for i in range(1, len(sample)):
            if sample[i] != sample[i - 1]:
                has_variation = True
                break

        if has_variation:
            varying_indices.append(idx)

    return varying_indices


if __name__ == '__main__':
    # 读取所有数据集
    oridata = load_data(oridata_path)
    ours_data = load_data(ours_path)
    diffts_data = load_data(diffts_path)
    testdata = load_data(testdata_path)

    # 假设负荷数据在第一列
    oridata_load = oridata[:, :, 0]  # 原始数据负荷列
    ours_load = ours_data[:, :, 0]  # ours数据负荷列
    diffts_load = diffts_data[:, :, 0]  # diffTS数据负荷列
    testdata_load = testdata[:, :, 0]  # 测试数据负荷列

    # 筛选具有变化的样本
    varying_indices = filter_varying_samples(oridata_load)
    print(f"Found {len(varying_indices)} samples with variations in original data")

    # 确保有足够的非恒定样本进行处理
    if len(varying_indices) < 4:
        print("Warning: Only found", len(varying_indices), "samples with variations")
        selected_indices = varying_indices[:len(varying_indices)]
    else:
        # 随机选择4个具有变化的样本
        selected_indices = np.random.choice(varying_indices, 16, replace=False)
        print("Selected varying sample indices:", selected_indices)

    # 处理筛选后的样本
    for count, i in enumerate(selected_indices):
        print(f"Processing sample {count + 1}/{len(selected_indices)} (Index: {i})...")
        oridata_sample = oridata_load[i]  # 选择有变化的样本

        # 分别从各数据集中寻找最相似的样本
        ours_similar, _ = find_most_similar_load(oridata_sample, ours_load, num_samples=5)
        diffts_similar, _ = find_most_similar_load(oridata_sample, diffts_load, num_samples=5)

        # 新增：从testdata中选择两个与原始数据最相似的样本
        testdata_similar, test_indices = find_most_similar_load(oridata_sample, testdata_load, num_samples=3)
        ours_similar, _ = find_most_similar_load2(testdata_similar, ours_load, num_samples=5)

        print(f"Selected testdata indices for comparison: {test_indices}")

        # 绘制对比图
        plot_comparison(oridata_sample, ours_similar, diffts_similar, testdata_similar, i)