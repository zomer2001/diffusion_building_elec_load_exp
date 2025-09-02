import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 设置设备（明确指定 CPU）
device = torch.device('cuda')

# 定义稀疏率和数据长度
sparsity_rates = [90, 80, 70, 50, 40, 30, 20]
lengths = [2160]
methods = ['oridata', 'wgan', 'diffts', 'diffts-fft', 'timegan', 'cgan']
base_dir = '../fakedata'
test_data_folder = '../testdata'

# 评估指标函数
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if np.sum(non_zero) == 0:
        return np.mean(np.abs((y_true - y_pred)))
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

# 定义 RNN 模型（替换原来的LSTM）
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.rnn(x)  # RNN替换LSTM
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 数据预处理：用6个特征的前23步预测第一个特征的第24步
def prepare_data(data, time_steps=24):
    X, y = [], []
    print(f"Input data shape: {data.shape}")  # Debug: print input shape
    for i in range(len(data) - time_steps + 1):
        x = data[i, :time_steps - 1, :]  # 前23步，所有6个特征，形状 (23, 6)
        X.append(x)
        y.append(data[i, time_steps - 1, 0])  # 第24步的第一个特征
    X = np.array(X)  # 形状: (samples, 23, 6)
    y = np.array(y)  # 形状: (samples,)
    print(f"X shape: {X.shape}, y shape: {y.shape}")  # Debug: final shapes
    return X, y

# 保存训练曲线
def save_training_curve(history, building_name, sparsity, method, output_dir='results/rnn/loss'):
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

# 主实验逻辑
results_all = []
for test_folder in os.listdir(test_data_folder):
    parts = test_folder.split('_')
    if len(parts) >= 4:
        building_name = '_'.join(parts[:-1])  # 修正：排除 length 和 sparsity
        length = int(parts[-2])
        sparsity = int(parts[-1])

        if length == 2160 and sparsity in sparsity_rates:
            # 读取测试集数据
            test_file = os.path.join(test_data_folder, test_folder, 'samples', 'energy_norm_truth_24_test.npy')
            if not os.path.exists(test_file):
                print(f"Test file {test_file} not found, skipping...")
                continue
            test_data = np.load(test_file)
            if test_data.shape[1:] != (24, 6):
                print(f"Invalid shape for {test_file}: {test_data.shape}, skipping...")
                continue
            X_test, y_test = prepare_data(test_data)
            X_test = torch.tensor(X_test, dtype=torch.float32).to(device)  # 形状: (samples, 23, 6)
            y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

            # 读取 oridata（训练数据）
            oridata_file = os.path.join(test_data_folder, test_folder, 'samples', 'energy_norm_truth_24_train.npy')
            if not os.path.exists(oridata_file):
                print(f"Oridata file {oridata_file} not found, skipping...")
                continue
            oridata = np.load(oridata_file)
            if oridata.shape[1:] != (24, 6):
                print(f"Invalid shape for {oridata_file}: {oridata.shape}, skipping...")
                continue
            X_oridata, y_oridata = prepare_data(oridata)
            X_oridata = torch.tensor(X_oridata, dtype=torch.float32).to(device)
            y_oridata = torch.tensor(y_oridata, dtype=torch.float32).to(device)

            # 遍历每种方法
            for method in methods:
                print(f"Training for {building_name}, sparsity {sparsity}, method {method}...")

                # 准备训练数据
                if method == 'oridata':
                    X_train, y_train = X_oridata, y_oridata
                else:
                    # 读取合成数据
                    X_train, y_train = X_oridata, y_oridata  # 默认使用 oridata
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
                                                continue
                                            X_synth, y_synth = prepare_data(synth_data)
                                            X_synth = torch.tensor(X_synth, dtype=torch.float32).to(device)
                                            y_synth = torch.tensor(y_synth, dtype=torch.float32).to(device)
                                            X_train = torch.cat([X_oridata, X_synth], dim=0)
                                            y_train = torch.cat([y_oridata, y_synth], dim=0)
                                            break
                    elif method in ['timegan', 'cgan', 'wgan']:
                        sparsity_folder = os.path.join(base_dir, method, str(sparsity))
                        file_name = f'generated_{building_name}.npy'
                        file_path = os.path.join(sparsity_folder, 'train', file_name)
                        if os.path.exists(file_path):
                            synth_data = np.load(file_path)
                            if synth_data.shape[1:] != (24, 6):
                                print(f"Invalid shape for {file_path}: {synth_data.shape}, skipping...")
                                continue
                            X_synth, y_synth = prepare_data(synth_data)
                            X_synth = torch.tensor(X_synth, dtype=torch.float32).to(device)
                            y_synth = torch.tensor(y_synth, dtype=torch.float32).to(device)
                            X_train = torch.cat([X_oridata, X_synth], dim=0)
                            y_train = torch.cat([y_oridata, y_synth], dim=0)
                        else:
                            print(f"Synthetic data {file_path} not found, skipping...")
                            continue

                # 设置不同方法的 epochs
                if method == 'oridata':
                    num_epochs = 2
                elif method == 'diffts-fft':
                    num_epochs = 5
                elif method == 'diffts':
                    num_epochs = 2
                else:  # timegan, cgan
                    num_epochs = 2

                # 初始化模型、损失函数和优化器（使用RNNModel）
                model = RNNModel(input_size=6).to(device)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                # 训练模型
                history = []
                batch_size = 32
                with tqdm(total=num_epochs, desc=f"Training {method}") as pbar:
                    for epoch in range(num_epochs):
                        model.train()
                        total_loss = 0
                        for i in range(0, len(X_train), batch_size):
                            batch_X = X_train[i:i + batch_size]
                            batch_y = y_train[i:i + batch_size]
                            optimizer.zero_grad()
                            outputs = model(batch_X).squeeze()
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item() * batch_X.size(0)
                        epoch_loss = total_loss / len(X_train)
                        history.append(epoch_loss)
                        pbar.update(1)

                # 保存训练曲线
                #save_training_curve(history, building_name, sparsity, method)

                # 测试模型
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_test).squeeze().cpu().numpy()
                y_test_np = y_test.cpu().numpy()
                metrics = calculate_metrics(y_test_np, y_pred)
                metrics['Building'] = building_name
                metrics['Sparsity'] = sparsity
                metrics['Method'] = method
                metrics['Epochs'] = num_epochs  # 记录使用的 epochs
                results_all.append(metrics)
                print(f"Metrics for {building_name}, sparsity {sparsity}, method {method}: {metrics}")

# 保存所有结果到总 CSV
if results_all:
    output_dir = '../results/rnn_2160'  # 修改输出目录名
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results_df = pd.DataFrame(results_all)
    results_df.to_csv(os.path.join(output_dir, 'all_results.csv'), index=False)
    print(f"Saved all results to {os.path.join(output_dir, 'all_results.csv')}")