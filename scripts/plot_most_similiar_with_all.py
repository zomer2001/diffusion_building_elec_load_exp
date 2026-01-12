import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# =========================
# Backend & Style
# =========================
matplotlib.use('TkAgg')

def set_academic_style():
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 15,
        'axes.titlesize': 17,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'lines.linewidth': 2.3,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'figure.facecolor': 'white'
    })


# =========================
# Path Configuration
# =========================
base_dir = '../fakedata'
test_data_folder = '../testdata'
result_root = './result'
sparsity_rates = [70]


# =========================
# Utility Functions
# =========================
def smooth_curve(x, window=15):
    if window <= 1:
        return x
    return np.convolve(x, np.ones(window) / window, mode='same')


def filter_varying_samples(data_load):
    valid_idx = []
    for i, s in enumerate(data_load):
        if np.all(s == 0):
            continue
        if np.all(s == s[0]):
            continue
        if np.any(np.diff(s) != 0):
            valid_idx.append(i)
    return valid_idx


def find_most_similar_load(target_load, candidates_load, num_samples=5):
    diff = np.abs(candidates_load - target_load)
    total_diff = np.sum(diff, axis=1)
    idx = np.argsort(total_diff)[:num_samples]
    return candidates_load[idx]


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
        if data:
            return np.concatenate(data, axis=0)
    return None


# =========================
# Visualization
# =========================
def plot_distribution_comparison(
    reference,
    test_group,
    gen_group,
    method_name,
    save_path
):
    set_academic_style()

    time_axis = np.arange(len(reference))

    ref_s = smooth_curve(reference)
    test_s = np.array([smooth_curve(t) for t in test_group])
    gen_s = np.array([smooth_curve(g) for g in gen_group])

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Test distribution
    ax.fill_between(
        time_axis,
        test_s.min(axis=0),
        test_s.max(axis=0),
        color='#1f77b4',
        alpha=0.18,
        label='Test Data Distribution'
    )

    # Generated distribution
    ax.fill_between(
        time_axis,
        gen_s.min(axis=0),
        gen_s.max(axis=0),
        color='#2ca02c' if method_name == 'Ours' else '#ff7f0e',
        alpha=0.22,
        label=f'{method_name} Generated Distribution'
    )

    # Reference curve
    ax.plot(
        time_axis,
        ref_s,
        color='#2b2b2b',
        linewidth=3.2,
        label='Training Reference'
    )

    ax.set_xlabel('Time (hour)')
    ax.set_ylabel('Normalized Load')
    ax.set_title(f'Distribution Comparison: Test vs {method_name}')
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# Main Loop
# =========================
processed_buildings = set()

for test_folder in os.listdir(test_data_folder):
    parts = test_folder.split('_')
    if len(parts) < 4:
        continue

    building_name = '_'.join(parts[:-1])
    length = int(parts[-2])
    sparsity = int(parts[-1])

    key = f'{building_name}_{sparsity}'
    if key in processed_buildings or length != 2160 or sparsity not in sparsity_rates:
        continue

    processed_buildings.add(key)
    print(f'Processing: {building_name}')

    # result subfolder
    save_dir = os.path.join(result_root, building_name)
    os.makedirs(save_dir, exist_ok=True)

    # load data
    oridata = np.load(
        os.path.join(test_data_folder, test_folder, 'samples', 'energy_norm_truth_24_train.npy')
    )[:, :, 0]

    testdata = np.load(
        os.path.join(test_data_folder, test_folder, 'samples', 'energy_norm_truth_24_test.npy')
    )[:, :, 0]

    ours_data = load_generated_data(
        os.path.join(base_dir, 'ours_gen', str(sparsity), building_name)
    )
    diffts_data = load_generated_data(
        os.path.join(base_dir, 'diffts-fft', str(sparsity), building_name)
    )

    if ours_data is None or diffts_data is None:
        continue

    ours_load = ours_data[:, :, 0]
    diffts_load = diffts_data[:, :, 0]

    valid_indices = filter_varying_samples(oridata)
    if len(valid_indices) < 3:
        continue

    selected_indices = np.random.choice(valid_indices, 3, replace=False)

    for idx in selected_indices:
        ref = oridata[idx]

        test_similar = find_most_similar_load(ref, testdata, 5)
        ours_similar = find_most_similar_load(ref, ours_load, 5)
        diffts_similar = find_most_similar_load(ref, diffts_load, 5)

        plot_distribution_comparison(
            ref, test_similar, ours_similar,
            'Ours',
            os.path.join(save_dir, f'{building_name}_idx{idx}_ours.png')
        )

        plot_distribution_comparison(
            ref, test_similar, diffts_similar,
            'DiffTS',
            os.path.join(save_dir, f'{building_name}_idx{idx}_diffts.png')
        )

print('All figures saved successfully.')
