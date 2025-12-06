
# HyperHand (SHREC) 训练、测试与三流融合指南

本指南旨在指导如何在 SHREC 数据集上训练 Joint、Bone 和 Motion 三个流的模型，并通过融合（Ensemble）达到最佳准确率。


-----

## 🚀 1. 单流训练 (Training)

我们需要分别训练三个流的模型。推荐使用以下通用超参数配置：

  * **Optimizer**: AdamW
  * **Base LR**: 0.0002
  * **Weight Decay**: 0.05
  * **Epoch**: 80
  * **Step**: [45, 60]
  * **Normalization**: **False** (非常重要！)

### 1.1 Joint 流 (关节流)

捕捉手势的绝对空间位置特征。

  * **配置文件**: `config/SHREC/supervised/hyperhand_supervised_SHREC_joint.yaml`
  * **关键参数**:
    ```yaml
    train_feeder_args:
      bone: False
      vel: False
      normalization: False
    ```
  * **运行命令**:
    ```bash
    python main.py finetune_evaluation --config config/SHREC/supervised/hyperhand_supervised_SHREC_joint.yaml
    ```

### 1.2 Bone 流 (骨骼流)

捕捉骨骼长度和方向的相对变化。

  * **配置文件**: `config/SHREC/supervised/hyperhand_supervised_SHREC_bone.yaml`
  * **关键参数**:
    ```yaml
    train_feeder_args:
      bone: True
      vel: False
      normalization: False
    ```
  * **运行命令**:
    ```bash
    python main.py finetune_evaluation --config config/SHREC/supervised/hyperhand_supervised_SHREC_bone.yaml
    ```

### 1.3 Motion 流 (运动流)

捕捉动作的速度和方向（帧差）。

  * **配置文件**: `config/SHREC/supervised/hyperhand_supervised_SHREC_motion.yaml`
  * **关键参数**:
    ```yaml
    train_feeder_args:
      bone: False
      vel: True
      normalization: False
    ```
  * **运行命令**:
    ```bash
    python main.py finetune_evaluation --config config/SHREC/supervised/hyperhand_supervised_SHREC_motion.yaml
    ```

-----

## 🧪 2. 生成最佳测试结果 (Testing)

训练结束后，默认的 `test_result.pkl` 是最后一个 Epoch 的结果（可能已过拟合）。必须重新加载 `best_model.pt` 来生成最佳结果。

**操作步骤（对每个流都要做一遍）：**

1.  **修改配置文件**：
      * 将 `phase` 改为 `test`。
      * 将 `weights` 指向对应的 `best_model.pt`。
2.  **运行测试命令**。
3.  **重命名结果文件**（防止被覆盖）。

**示例命令：**

**处理Joint流**
1. 运行测试 (确保配置文件里已设为 phase: test 和 weights: .../best_model.pt)
```bash

python main.py finetune_evaluation --config config/SHREC/supervised/hyperhand_supervised_SHREC_joint_test.yaml
```
2.重命名
```bash
mv work_dir/SHREC/hyperhand_joint/test_result.pkl work_dir/SHREC/hyperhand_joint/best_result.pkl
```
**处理 Bone 流**
```bash
python main.py finetune_evaluation --config config/SHREC/supervised/hyperhand_supervised_SHREC_bone.yaml
```
```bash
mv work_dir/SHREC/hyperhand_bone/test_result.pkl work_dir/SHREC/hyperhand_bone/best_result.pkl
```
**处理 Motion 流**
```bash
python main.py finetune_evaluation --config config/SHREC/supervised/hyperhand_supervised_SHREC_motion_test.yaml
```
```bash
mv work_dir/SHREC/hyperhand_motion/test_result.pkl work_dir/SHREC/hyperhand_motion/best_result.pkl
```

-----

## 🔗 3. 三流融合 (Ensemble)

```bash
python ensemble_shrec.py
```

**预期效果**：

  * **Joint 单流**: \~79%
  * **Joint + Bone**: \~84%
  * **Joint + Bone + Motion**: **\> 86%** (甚至可能达到 90%)