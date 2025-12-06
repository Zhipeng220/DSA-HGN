import pickle
import numpy as np
from tqdm import tqdm
import os

print("Starting Three-Stream Ensemble Evaluation (Joint + Bone + Motion)...")

# --- 1. 配置路径 (请确保 Bone 的路径正确) ---

# Joint 结果 (Stream 1)
joint_path = './work_dir/SHREC/hyperhand_joint/test_result.pkl'

# Bone 结果 (Stream 2) - *新增*
# 假设你的 bone 模型保存在 hyperhand_bone 文件夹下，请根据实际情况修改
bone_path = './work_dir/SHREC/hyperhand_bone/test_result.pkl'

# Motion 结果 (Stream 3)
motion_path = './work_dir/SHREC/hyperhand_motion/test_result.pkl'

# 标签文件路径
label_path = '/Users/gzp/Desktop/exp/DATA/SHREC2017_data/val_label.pkl'

# --- 2. 融合策略 ---
# 三流融合通常策略：
# Joint 和 Bone 提供极强的空间特征，Motion 提供互补的时间特征。
# 常用权重设置:
# A. [1.0, 1.0, 1.0] (简单平均)
# B. [2.0, 2.0, 1.0] (强调空间结构，降低运动流噪声)
# C. [1.0, 1.0, 0.8] (推荐起点)

alpha = [1.0, 0.6, 0.8]

print(f"Fusion Weights -> Joint: {alpha[0]}, Bone: {alpha[1]}, Motion: {alpha[2]}")


# ----------------------------------------

def load_pkl(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        # 尝试查找 best_result.pkl 作为备选
        alt_path = path.replace('test_result.pkl', 'best_result.pkl')
        if os.path.exists(alt_path):
            print(f"Found alternative file: {alt_path}")
            with open(alt_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Cannot find result file at {path}")

    with open(path, 'rb') as f:
        return pickle.load(f)


try:
    # 1. 加载 Joint
    print(f"Loading Joint results from {joint_path}...")
    r1_dict = load_pkl(joint_path)

    # 2. 加载 Bone
    print(f"Loading Bone results from {bone_path}...")
    r2_dict = load_pkl(bone_path)

    # 3. 加载 Motion
    print(f"Loading Motion results from {motion_path}...")
    r3_dict = load_pkl(motion_path)

    # 4. 加载标签
    print(f"Loading labels from {label_path}...")
    with open(label_path, 'rb') as f:
        label_data = pickle.load(f)
        if isinstance(label_data, tuple) or isinstance(label_data, list):
            sample_names = label_data[0]
            true_labels = label_data[1]
        elif isinstance(label_data, dict):
            sample_names = list(label_data.keys())
            true_labels = list(label_data.values())
        else:
            raise ValueError("Unknown label file format")

    right_num = total_num = right_num_5 = 0

    # 遍历所有样本
    for i in tqdm(range(len(sample_names))):
        name = sample_names[i]
        label = int(true_labels[i])

        # 检查样本是否在所有三个流中都存在
        if name not in r1_dict:
            print(f"Warning: Sample {name} missing in Joint.")
            continue
        if name not in r2_dict:
            print(f"Warning: Sample {name} missing in Bone.")
            continue
        if name not in r3_dict:
            print(f"Warning: Sample {name} missing in Motion.")
            continue

        r1 = r1_dict[name]  # Joint Score
        r2 = r2_dict[name]  # Bone Score
        r3 = r3_dict[name]  # Motion Score

        # --- 核心融合步骤 (三流) ---
        # Result = w1*J + w2*B + w3*M
        r = r1 * alpha[0] + r2 * alpha[1] + r3 * alpha[2]
        # -------------------------

        # Top-1 Accuracy
        pred = np.argmax(r)
        if pred == label:
            right_num += 1

        # Top-5 Accuracy
        rank_5 = r.argsort()[-5:]
        if label in rank_5:
            right_num_5 += 1

        total_num += 1

    if total_num > 0:
        acc = right_num / total_num
        acc5 = right_num_5 / total_num

        print('-' * 60)
        print(f'Final Fusion Accuracy (Joint + Bone + Motion)')
        print(f'Weights: Joint={alpha[0]}, Bone={alpha[1]}, Motion={alpha[2]}')
        print('-' * 60)
        print(f'Top-1 Accuracy: {acc * 100:.2f}%')
        print(f'Top-5 Accuracy: {acc5 * 100:.2f}%')
        print('-' * 60)
    else:
        print("No common samples found to evaluate.")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("建议检查: 1. Bone 流的路径是否正确; 2. 所有 pickle 文件是否完整。")

# ... (保留你之前的加载代码，确保 r1, r2, r3 已经加载进内存) ...

print("\n" + "=" * 20 + " 开始自动权重搜索 (Grid Search) " + "=" * 20)

best_acc = 0
best_params = []

# 定义搜索范围：0.1 到 2.0，步长 0.2
weights_range = [i / 10.0 for i in range(2, 21, 2)]

# 为了加速，固定 Joint = 1.0，只搜索 Bone 和 Motion 的相对比例
# 因为 [1, 1, 1] 和 [2, 2, 2] 的结果是一样的，只看比例
for w_bone in weights_range:
    for w_motion in weights_range:
        current_right = 0

        # 向量化计算 (比 for 循环快 100 倍)
        # 假设 r1_dict 等已转换为 list 或 numpy array 顺序一致
        # 这里为了演示逻辑，仍使用简化的循环，实际使用建议先转 numpy
        for i in range(len(sample_names)):
            name = sample_names[i]
            label = int(true_labels[i])

            # 获取分数
            s1 = r1_dict[name]
            s2 = r2_dict[name]
            s3 = r3_dict[name]

            # 融合
            score = s1 * 1.0 + s2 * w_bone + s3 * w_motion

            if np.argmax(score) == label:
                current_right += 1

        curr_acc = current_right / total_num

        if curr_acc > best_acc:
            best_acc = curr_acc
            best_params = [1.0, w_bone, w_motion]
            print(f"New Best! Acc: {best_acc * 100:.2f}% | Weights: {best_params}")

print(f"\nFinal Best Accuracy: {best_acc * 100:.2f}%")
print(f"Best Weights -> Joint: 1.0, Bone: {best_params[1]}, Motion: {best_params[2]}")