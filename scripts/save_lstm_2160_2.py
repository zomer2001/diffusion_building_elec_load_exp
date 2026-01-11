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

# ================= 设备 =================
device = torch.device('mps')

# ================= 参数配置 =================
sparsity_rates = [90, 80, 70, 50, 40, 30, 20]
lengths = [2160]
methods = ['ours','']

base_dir = '../fakedata'
test_data_folder = '../testdata'


# ================= 指标函数 =================
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if np.sum(non_zero) == 0:
        return np.mean(np.abs(y_true - y_pred))
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}


# ================= LSTM 模型 =================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        return self.fc2(out)


# ================= 数据预处理 =================
def prepare_data(data, time_steps=24):
    X, y = [], []
    for i in range(len(data) - time_steps + 1):
        X.append(data[i, :time_steps - 1, :])      # (23, 6)
        y.append(data[i, time_steps - 1, 0])       # 第 24 步第 1 个特征
    return np.array(X), np.array(y)


# ================= 主实验 =================
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

    # ---------- test data ----------
    test_file = os.path.join(
        test_data_folder, test_folder, 'samples',
        'energy_norm_truth_24_test.npy'
    )
    if not os.path.exists(test_file):
        continue

    test_data = np.load(test_file)
    if test_data.shape[1:] != (24, 6):
        continue

    X_test, y_test = prepare_data(test_data)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # ---------- oridata ----------
    oridata_file = os.path.join(
        test_data_folder, test_folder, 'samples',
        'energy_norm_truth_24_train.npy'
    )
    if not os.path.exists(oridata_file):
        continue

    oridata = np.load(oridata_file)
    if oridata.shape[1:] != (24, 6):
        continue

    X_oridata, y_oridata = prepare_data(oridata)
    X_oridata = torch.tensor(X_oridata, dtype=torch.float32).to(device)
    y_oridata = torch.tensor(y_oridata, dtype=torch.float32).to(device)

    print(f"\n[OK] {building_name} | sparsity {sparsity}")

    # ---------- 方法循环 ----------
    for method in methods:
        print(f"→ Method: {method}")

        X_train, y_train = X_oridata, y_oridata
        synth_loaded = False

        # ===== test-driven 合成数据读取 =====
        if method in ['ours', 'diffts']:
            sparsity_folder = os.path.join(
                base_dir, method, str(sparsity), building_name
            )

            if not os.path.exists(sparsity_folder):
                print("  [Skip] No synthetic folder")
                continue

            for sub in os.listdir(sparsity_folder):
                sub_path = os.path.join(sparsity_folder, sub)
                if not os.path.isdir(sub_path):
                    continue

                for f in os.listdir(sub_path):
                    if f.endswith('.npy'):
                        file_path = os.path.join(sub_path, f)
                        synth_data = np.load(file_path)
                        if synth_data.shape[1:] != (24, 6):
                            continue

                        X_s, y_s = prepare_data(synth_data)
                        X_s = torch.tensor(X_s, dtype=torch.float32).to(device)
                        y_s = torch.tensor(y_s, dtype=torch.float32).to(device)

                        X_train = torch.cat([X_oridata, X_s], dim=0)
                        y_train = torch.cat([y_oridata, y_s], dim=0)
                        synth_loaded = True
                        break

                if synth_loaded:
                    break

            if not synth_loaded:
                print("  [Skip] No valid synthetic file")
                continue

        # ===== 训练参数 =====
        num_epochs = 8 if method == 'ours' else 2
        batch_size = 32

        model = LSTMModel(input_size=6).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # ===== 训练 =====
        for _ in tqdm(range(num_epochs), desc=f"Training {method}", leave=False):
            model.train()
            for i in range(0, len(X_train), batch_size):
                bx = X_train[i:i + batch_size]
                by = y_train[i:i + batch_size]
                optimizer.zero_grad()
                loss = criterion(model(bx).squeeze(), by)
                loss.backward()
                optimizer.step()

        # ===== 测试 =====
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).squeeze().cpu().numpy()

        metrics = calculate_metrics(y_test.cpu().numpy(), y_pred)
        metrics.update({
            'Building': building_name,
            'Sparsity': sparsity,
            'Method': method,
            'Epochs': num_epochs
        })

        results_all.append(metrics)
        print("  Metrics:", metrics)


# ================= 保存结果 =================
if results_all:
    output_dir = '../results/lstm_2160_2_new_0111'
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results_all)
    csv_path = os.path.join(output_dir, 'all_results.csv')
    df.to_csv(csv_path, index=False)

    print(f"\n[Saved] {csv_path}")
