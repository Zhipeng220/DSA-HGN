import pickle
import numpy as np
import os

print("=== Re-running Grid Search with ALL BEST files ===")

# --- 1. 确保全部加载 best_result ---
joint_path = './work_dir/SHREC/hyperhand_joint/best_result.pkl'
bone_path = './work_dir/SHREC/hyperhand_bone/best_result.pkl'  # 确保这里存在
motion_path = './work_dir/SHREC/hyperhand_motion/best_result.pkl'
label_path = '/Users/gzp/Desktop/exp/DATA/SHREC2017_data/val_label.pkl'


def load_pkl(path):
    # 强制确保加载的是 best
    if 'test_result' in path:
        path = path.replace('test_result', 'best_result')
    print(f"Loading: {path}")
    return pickle.load(open(path, 'rb'))


try:
    r1_dict = load_pkl(joint_path)
    r2_dict = load_pkl(bone_path)
    r3_dict = load_pkl(motion_path)

    with open(label_path, 'rb') as f:
        label_data = pickle.load(f)
        sample_names = label_data[0]
        true_labels = label_data[1]

    total_num = len(sample_names)
    best_acc = 0
    best_weights = []

    # --- 2. 扩大搜索策略 ---
    # 既然 Bone 变强了，我们需要给它更高的搜索空间
    # 搜索范围: 0.2 到 2.0，步长 0.2

    # 这里的逻辑是: Joint 固定 1.0, 搜索 Bone 和 Motion 的比例
    for w_bone in [i / 10.0 for i in range(2, 25, 2)]:  # 0.2 - 2.4
        for w_motion in [i / 10.0 for i in range(2, 21, 2)]:  # 0.2 - 2.0

            right = 0
            for i in range(total_num):
                name = sample_names[i]
                if name not in r1_dict: continue

                label = int(true_labels[i])

                # 融合
                # 注意：Joint 固定为 1.0
                score = r1_dict[name] * 1.0 + \
                        r2_dict[name] * w_bone + \
                        r3_dict[name] * w_motion

                if np.argmax(score) == label:
                    right += 1

            acc = right / total_num

            if acc > best_acc:
                best_acc = acc
                best_weights = [1.0, w_bone, w_motion]
                print(f"New Best: {best_acc * 100:.2f}% | Weights: {best_weights}")

    print("\n" + "=" * 40)
    print(f"FINAL RESULT (All Best Files)")
    print(f"Max Accuracy: {best_acc * 100:.2f}%")
    print(f"Optimal Weights: Joint=1.0, Bone={best_weights[1]}, Motion={best_weights[2]}")
    print("=" * 40)

except Exception as e:
    print(f"Error: {e}")