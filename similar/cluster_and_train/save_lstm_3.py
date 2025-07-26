import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import kl_div
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

def sliding_window(data, window_size=24):
    """
    使用滑动窗口生成数据样本
    参数：
        data: 输入的二维数据数组，形状为 [n_samples, n_features]
        window_size: 窗口大小（默认为24）
    返回：
        通过滑动窗口生成的三维数组，每个样本的形状为 [window_size, n_features]
    """
    samples = []
    for i in range(len(data) - window_size + 1):
        samples.append(data[i:i + window_size])  # 滑动窗口提取样本
    return np.array(samples)

def save_as_npy(file_path, data, output_dir, i):
    """
    将数据保存为npy文件
    参数：
        file_path: 输入文件的路径
        data: 要保存的数据（需要为二维数组）
        output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 构造输出文件路径
    file_name = os.path.splitext(os.path.basename(file_path))[0] + '.npy'
    npy_file_path = os.path.join(output_dir, file_name)

    np.save(f'{npy_file_path}_{i}', data)  # 加上索引以避免文件覆盖
    print(f"Saved {file_name} to {npy_file_path}_{i}.npy")

# 设置输入文件路径和输出目录
building_test_name = 'Hog_public_Kevin_start2000_720'
building_test_name1 = 'Hog_public_Kevin'
output_plot = 'fixed_training_results_5.csv'
csv_file_path = f'/Users/zomeryang/Documents/bearlab/keti/project/data_prepare/standard_selected/{building_test_name1}.csv'  # 替换为CSV文件路径
output_dir = f'/Users/zomeryang/Documents/bearlab/keti/project/exp/similiar/cluster_and_train/make_test/output_npy_files/{building_test_name1}'  # 输出目录

# 从CSV文件读取数据
df = pd.read_csv(csv_file_path)

# 假设有多个特征，我们需要使用所有特征（例如：负荷、气温等）
# 假设 CSV 文件中前 6 列是我们需要的特征
data = df.iloc[:, :6].values  # 获取前6列数据，形状 [n_samples, 6]

# 归一化处理：使用MinMaxScaler对每一列进行归一化
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)  # 归一化数据

# 使用滑动窗口将数据转换为24步的样本
window_size = 24
sliding_data = sliding_window(data_normalized, window_size)

# 每个npy文件保存2160个样本（如果数据量足够的话）
n_samples_per_npy = 1080
num_files = len(sliding_data) // n_samples_per_npy  # 确保样本数可以被2160整除

for i in range(num_files):
    start_idx = i * n_samples_per_npy
    end_idx = start_idx + n_samples_per_npy
    samples = sliding_data[start_idx:end_idx]  # 获取每2160个样本

    # 保存为npy文件
    save_as_npy(csv_file_path, samples, output_dir, i)

print(f"Total {num_files} .npy files have been saved.")

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 只考虑稀疏率50
sparsity_rates = [50]
lengths = [2160]
methods = ['oridata', 'ours', 'diffts', 'timegan', 'cgan']
base_dir = '/Users/zomeryang/Documents/bearlab/keti/project/exp/24_real_exp/fakedata'
test_data_folder = f'/Users/zomeryang/Documents/bearlab/keti/project/exp/similiar/cluster_and_train/make_test/output_npy_files/{building_test_name1}'

# 评估指标函数
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if np.sum(non_zero) == 0:
        return np.mean(np.abs((y_true - y_pred)))
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


def calculate_mmd(oridata, testdata, kernel='rbf', gamma=None):
    """
    计算最大平均差异(MMD)作为分布相似度的度量

    参数:
    oridata: 原始数据集 (形状: [n_samples, seq_len])
    testdata: 测试数据集 (形状: [m_samples, seq_len])
    kernel: 核函数类型 ('rbf', 'linear', 'poly')
    gamma: RBF核的带宽参数

    返回:
    mmd值: 值越小表示两个分布越相似
    """
    X = torch.tensor(oridata, dtype=torch.float32)
    Y = torch.tensor(testdata, dtype=torch.float32)

    n = X.shape[0]
    m = Y.shape[0]

    # 自动设置gamma (经验法则: 1 / 特征数)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    # 计算核矩阵
    def kernel_matrix(Z1, Z2):
        if kernel == 'rbf':
            # RBF核: k(x,y) = exp(-gamma * ||x-y||^2)
            norm = torch.cdist(Z1, Z2) ** 2
            return torch.exp(-gamma * norm)
        elif kernel == 'linear':
            # 线性核: k(x,y) = <x,y>
            return torch.mm(Z1, Z2.t())
        elif kernel == 'poly':
            # 多项式核: k(x,y) = (<x,y> + 1)**2
            return (torch.mm(Z1, Z2.t()) + 1) ** 2
        else:
            raise ValueError(f"未知的核函数: {kernel}")

    # 计算三项核均值
    K_XX = kernel_matrix(X, X)
    K_YY = kernel_matrix(Y, Y)
    K_XY = kernel_matrix(X, Y)

    # 计算MMD²
    mmd2 = (K_XX.sum() / (n * (n - 1)) +
            K_YY.sum() / (m * (m - 1)) -
            2 * K_XY.sum() / (n * m))

    # 处理数值稳定性问题
    mmd2 = torch.max(mmd2, torch.tensor(0.0))

    return torch.sqrt(mmd2).item()


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}


# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def prepare_data(data, time_steps=24):
    X, y = [], []
    for i in range(len(data) - time_steps + 1):
        x = data[i, :time_steps - 1, :]
        X.append(x)
        y.append(data[i, time_steps - 1, 0])
    return np.array(X), np.array(y)


def save_training_curve(history, building_name, sparsity, method, output_dir='results/lstm/loss'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Training Loss')
    plt.title(f'Training Loss Curve\nBuilding: {building_name}, Sparsity: {sparsity}, Method: {method}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    filename = os.path.join(output_dir, f'{building_name}_sparsity_{sparsity}_{method}_loss_curve.png')
    plt.savefig(filename)
    plt.close()
    print(f"Saved training curve to {filename}")


# 1. 首先训练固定模型
def load_synth_data(method, sparsity, building_name):
    if method in ['ours', 'diffts']:
        sparsity_folder = os.path.join(base_dir, method, str(sparsity), f'{building_test_name}',)
        for sub_folder in os.listdir(sparsity_folder):
            sub_folder_path = os.path.join(sparsity_folder, sub_folder)
            if os.path.isdir(sub_folder_path):
                for file in os.listdir(sub_folder_path):
                    if file.endswith('.npy'):
                        file_path = os.path.join(sub_folder_path, file)
                        synth_data = np.load(file_path)
                        if synth_data.shape[1:] == (24, 6):
                            return synth_data
    elif method in ['timegan', 'cgan']:
        file_path = f'/Users/zomeryang/Documents/bearlab/keti/project/exp/24_real_exp/fakedata/timegan/50/train/generated_{building_test_name}.npy'
        if os.path.exists(file_path):
            synth_data = np.load(file_path)
            if synth_data.shape[1:] == (24, 6):
                return synth_data
    return None


# 主逻辑
results_all = []

# 步骤1: 固定训练数据 - 选择第一个数据集作为训练数据
train_building = None
for test_folder in os.listdir(test_data_folder):
    if test_folder.endswith('_files') and train_building is None:
        parts = test_folder.split('_')
        building_name = '_'.join(parts[:-1])
        length = int(parts[-2])
        sparsity = int(parts[-1])
        train_building = building_name
        break

# if train_building is None:
#     raise ValueError("No suitable training data found")

print(f"Using fixed training data from building: {train_building}")

# 步骤2: 准备固定训练数据
#oridata_file = os.path.join(test_data_folder, f'{train_building}_2160_50', 'samples', 'energy_norm_truth_24_train.npy')
oridata = np.load(
    f'/Users/zomeryang/Documents/bearlab/keti/project/exp/24_real_exp/fakedata/diffts/50/{building_test_name}/{building_test_name}_50/samples/energy_norm_truth_24_train.npy')
X_train_fixed, y_train_fixed = prepare_data(oridata)
X_train_fixed = torch.tensor(X_train_fixed, dtype=torch.float32).to(device)
y_train_fixed = torch.tensor(y_train_fixed, dtype=torch.float32).to(device)

# 步骤3: 为每种方法准备训练模型
method_models = {}
method_epochs = {'oridata': 5, 'ours': 30, 'diffts': 30, 'timegan': 5, 'cgan': 5}

for method in methods:
    print(f"\n{'=' * 50}")
    print(f"Training fixed model for method: {method}")
    print(f"{'=' * 50}")

    # 准备训练数据
    if method == 'oridata':
        X_train = X_train_fixed
        y_train = y_train_fixed
    else:
        synth_data = load_synth_data(method, 50, train_building)
        if synth_data is None:
            print(f"Synthetic data for {method} not found, skipping...")
            continue

        X_synth, y_synth = prepare_data(synth_data)
        X_synth = torch.tensor(X_synth, dtype=torch.float32).to(device)
        y_synth = torch.tensor(y_synth, dtype=torch.float32).to(device)

        # 拼接原始数据和合成数据
        X_train = torch.cat([X_train_fixed, X_synth], dim=0)
        y_train = torch.cat([y_train_fixed, y_synth], dim=0)

    # 训练模型
    model = LSTMModel(input_size=6).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = []
    batch_size = 32
    num_epochs = method_epochs[method]

    with tqdm(total=num_epochs, desc=f"Training {method}") as pbar:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            indices = torch.randperm(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for i in range(0, len(X_train_shuffled), batch_size):
                end_idx = min(i + batch_size, len(X_train_shuffled))
                batch_X = X_train_shuffled[i:end_idx]
                batch_y = y_train_shuffled[i:end_idx]

                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_X.size(0)

            epoch_loss = total_loss / len(X_train_shuffled)
            history.append(epoch_loss)
            pbar.update(1)

    # 保存训练曲线
    save_training_curve(history, train_building, 50, method)
    method_models[method] = model

# 步骤4: 用所有测试数据集测试
test_cases = []
for test_folder in os.listdir(test_data_folder):
    if test_folder.endswith('.npy'):
        parts = test_folder.split('_')
        building_name = '_'.join(parts[:-1])
        length = test_folder
        sparsity = 5
        test_cases.append((building_name, length, sparsity))

print(f"\nFound {len(test_cases)} test datasets")

# 每个测试集包含多个文件 (_0.npy 到 _6.npy)
for building_name, length, sparsit in test_cases:
    print(f"\nTesting on building: {building_name}, sparsity: 5")

    # 准备测试数据
    test_data_all = []
    test_file = os.path.join(test_data_folder, length)
    # for i in range(7):  # _0.npy 到 _6.npy
    #     test_file = os.path.join(test_data_folder, f'Cockatoo_industrial_Nathaniel.npy_{i}.npy')
    #     if not os.path.exists(test_file):
    #         print(f"Test file not found: {test_file}")
    #         continue

    # 加载测试数据
    test_data = np.load(test_file, allow_pickle=True)

    print(f"Loaded test data shape: {test_data.shape}")
    test_data_all = test_data

    # 如果没有找到有效的测试文件，跳过该building
    # if not test_data_all:
    #     print(f"No valid test files found for {building_name}")
    #     continue

    # 将所有数据合并
    #test_data_all = np.concatenate(test_data_all, axis=0)  # 合并所有测试数据

    # 准备输入数据
    X_test, y_test = prepare_data(test_data_all)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    y_test_np = y_test.cpu().numpy()

    # 计算KL相似度
    oridata_load = oridata[:, :, 0]
    test_data_load = test_data_all[:, :, 0]
    similarity = calculate_mmd(oridata_load, test_data_load)

    # 对每个方法进行测试
    for method, model in method_models.items():
        print(f"Testing method: {method}")

        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).squeeze().cpu().numpy()

        # 计算评估指标
        metrics = calculate_metrics(y_test_np, y_pred)
        metrics['Building'] = building_name
        metrics['Sparsity'] = sparsity
        metrics['Method'] = method
        metrics['KL_Similarity'] = similarity
        metrics['Training_Building'] = train_building  # 记录使用的训练数据来源
        results_all.append(metrics)

        print(f"Metrics for {building_name} with {method}:")
        for k, v in metrics.items():
            if k != 'Building' and k != 'Method' and k != 'Sparsity' and k != 'KL_Similarity' and k != 'Training_Building':
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")

# 保存结果
if results_all:
    output_dir = 'results/lstm'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results_df = pd.DataFrame(results_all)
    csv_path = os.path.join(output_dir, output_plot)
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved all results to {csv_path}")
    print(f"Total test cases: {len(results_all)}")
else:
    print("No results to save")
