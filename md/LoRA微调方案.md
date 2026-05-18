# Agent-Assistant 项目 LoRA 微调方案（v2.0）

---

## 一、项目现状分析

### 1.1 当前实现状态 ✅

agent-assistant 项目**已实现完整的 LoRA 微调功能**：

| 功能模块 | 状态 | 文件位置 |
|---------|------|---------|
| LoRA 训练脚本 | ✅ 已实现 | `train_lora.py`, `train_lora_medical_terms.py` |
| LoRA 推理脚本 | ✅ 已实现 | `lora_inference.py`, `medical_terms_inference.py` |
| 应用集成 | ✅ 已实现 | `app.py`（支持本地/API双模式） |
| 工具集成 | ✅ 已实现 | `tools.py`（医疗术语解释器） |
| 数据集 | ✅ 已实现 | `data/medical_finetune.json` |

### 1.2 已支持的模型

| 模型 | 参数 | 状态 | 建议显存 |
|------|------|------|---------|
| GLM-4-6B-Chat | 6B | ✅ 已配置 | ≥8GB |
| GLM-4-9B-Chat | 9B | ⚠️ 可选 | ≥16GB |

### 1.3 引入 LoRA 微调的收益

| 维度 | API 调用方式 | LoRA 微调方式 |
|------|-------------|--------------|
| **成本** | 按调用次数计费 | 一次性训练成本 |
| **隐私** | 数据上传至第三方 | 数据本地化训练 |
| **定制化** | 依赖 Prompt Engineering | 针对领域数据微调 |
| **响应延迟** | 受网络影响 | 本地推理，延迟低 |
| **领域适配** | 通用模型，领域知识有限 | 可注入医疗领域知识 |

---

## 二、LoRA 原理深入解析

### 2.1 核心思想

LoRA（Low-Rank Adaptation）通过在 Transformer 模型的关键层插入低秩矩阵，实现参数高效微调：

```
原始权重: W ∈ R^{d×k}
LoRA 分解: W = W_0 + ΔW = W_0 + A × B
           其中 A ∈ R^{d×r}, B ∈ R^{r×k}, r << min(d,k)
```

### 2.2 数学原理

```python
# LoRA 前向传播（PEFT 库实现）
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # 低秩矩阵 A (随机初始化)
        self.A = nn.Parameter(torch.randn(in_dim, rank))
        # 低秩矩阵 B (初始化为0)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        
        # 缩放因子
        self.scaling = alpha / rank
    
    def forward(self, x):
        # 原始输出 + LoRA增量
        return x @ self.A @ self.B * self.scaling
```

### 2.3 参数效率对比

| 微调方式 | 训练参数量 | 存储需求 | 适用场景 |
|---------|-----------|---------|---------|
| 全参数微调 | 100% | 完整模型 | 资源充足、追求最佳效果 |
| LoRA (r=8) | ~0.1% | 仅存储 LoRA 权重 | 资源有限、领域适配 |
| LoRA (r=64) | ~1% | 少量额外存储 | 平衡效果与效率 |

---

## 三、LoRA 微调完整实现方案

### 3.1 环境准备

```bash
# 安装必要依赖
pip install torch torchvision transformers accelerate peft datasets evaluate
pip install sentencepiece  # GLM模型需要
pip install scikit-learn   # 评估指标
```

### 3.2 数据集准备

#### 3.2.1 数据格式要求

```json
[
    {
        "instruction": "解释什么是高血压",
        "input": "",
        "output": "高血压是指动脉血压持续升高，收缩压≥140mmHg和/或舒张压≥90mmHg..."
    }
]
```

#### 3.2.2 支持的数据集来源

| 数据集 | 来源 | 描述 |
|-------|------|------|
| 华图知识图谱 | `FreedomIntelligence/huatuo_knowledge_graph_qa` | 医疗问答数据集 |
| 本地数据集 | `data/medical_finetune.json` | 医疗术语解释数据 |

### 3.3 训练脚本使用指南

#### 3.3.1 通用训练脚本

```bash
# 使用通用数据集训练
python train_lora.py
```

#### 3.3.2 医疗术语专项训练

```bash
# 使用华图知识图谱数据集训练医疗术语解释模型
python train_lora_medical_terms.py
```

### 3.4 关键配置参数

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| **model_name** | `THUDM/glm-4-6b-chat` | 基础模型选择 |
| **lora_rank** | 8 | 低秩矩阵的秩 |
| **lora_alpha** | 16 | 缩放因子（通常为 rank 的 2 倍） |
| **batch_size** | 2-4 | 根据 GPU 显存调整 |
| **epochs** | 3-5 | 避免过拟合 |
| **learning_rate** | 3e-4 | LoRA 学习率通常较大 |

### 3.5 训练执行流程

```
1. 加载基础模型 → 2. 配置 LoRA → 3. 加载数据集 → 4. 执行训练 → 5. 保存权重
```

---

## 四、推理与集成

### 4.1 独立推理

```bash
# 通用推理
python lora_inference.py

# 医疗术语推理
python medical_terms_inference.py
```

### 4.2 集成到 Flask 应用

#### 4.2.1 配置方式

在 `.env` 文件中配置：

```env
# 使用 LoRA 本地模型
USE_LOCAL_MODEL=true
LORA_PATH=./lora_medical_terms

# 或使用 API 模式
USE_LOCAL_MODEL=false
OPENAI_API_KEY=your-api-key
MODEL=glm-5v-turbo
```

#### 4.2.2 启动应用

```bash
python app.py
```

### 4.3 工具集成

项目已将 LoRA 模型集成到医疗术语解释工具：

```python
# tools.py 中的集成
class MedicalTermExplainer:
    def __init__(self, base_model_name="THUDM/glm-4-6b-chat", lora_path="./lora_medical_terms"):
        # 加载基础模型和 LoRA 权重
        self.model = PeftModel.from_pretrained(base_model, lora_path)
    
    def explain(self, term):
        # 使用 LoRA 模型解释医疗术语
        ...
```

---

## 五、训练配置详解

### 5.1 LoRA 配置参数

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,      # 任务类型：因果语言模型
    r=8,                               # 低秩矩阵的秩
    lora_alpha=16,                     # 缩放因子
    target_modules=["query_key_value"], # 目标模块（GLM 的注意力层）
    lora_dropout=0.05,                 # Dropout 概率
    bias="none",                       # 不训练 bias
    inference_mode=False               # 训练模式
)
```

### 5.2 训练参数配置

```python
training_args = TrainingArguments(
    output_dir="./lora_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,      # 梯度累积
    learning_rate=3e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    fp16=True,                          # 混合精度训练
    report_to="none"
)
```

### 5.3 数据集处理

```python
class MedicalTermDataset(Dataset):
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建提示格式
        prompt = f"解释医疗术语：{item['instruction']}\n解释："
        full_text = prompt + item['output']
        
        # 编码
        encoding = self.tokenizer(full_text, ...)
        
        # 标签处理：只监督输出部分
        labels = encoding["input_ids"].clone()
        labels[0, :prompt_length] = -100  # prompt 部分忽略
        
        return {"input_ids": ..., "labels": ...}
```

---

## 六、最佳实践与调优建议

### 6.1 超参数调优指南

| 参数 | 范围 | 建议值 |
|------|------|-------|
| **rank (r)** | 4-64 | 8-16（医疗领域） |
| **alpha** | 8-64 | rank × 2 |
| **dropout** | 0.0-0.2 | 0.05 |
| **learning_rate** | 1e-5-1e-3 | 3e-4 |
| **epochs** | 1-20 | 3-5 |

### 6.2 训练技巧

1. **梯度累积**：小 batch size 时使用 `gradient_accumulation_steps=4`
2. **学习率调度**：使用线性衰减学习率
3. **混合精度**：开启 `fp16=True` 减少显存占用
4. **参数冻结**：确保只训练 LoRA 参数（PEFT 自动处理）

### 6.3 评估指标

```python
from evaluate import load

# BLEU 分数（机器翻译评估）
bleu = load("bleu")

# ROUGE 分数（文本摘要评估）
rouge = load("rouge")
```

### 6.4 常见问题与解决

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 显存不足 | 模型过大 | 使用 smaller 模型（如 6B） |
| 过拟合 | 训练数据少 | 增加数据量或使用 dropout |
| 收敛慢 | 学习率太低 | 增大学习率至 3e-4 |
| 推理速度慢 | 模型在 CPU 上运行 | 确保使用 GPU（device_map="auto"） |

---

## 七、项目扩展建议

### 7.1 LoRA 权重合并（部署优化）

```python
# 将 LoRA 权重合并到基础模型
model = model.merge_and_unload()
model.save_pretrained("./merged_model")
```

### 7.2 增量微调

```python
# 加载已有 LoRA 权重继续训练
model = PeftModel.from_pretrained(model, existing_lora_path)
```

### 7.3 多任务微调

```json
{
    "task_type": "medical_term",  # 任务类型标识
    "instruction": "...",
    "input": "...",
    "output": "..."
}
```

---

## 八、文件结构

```
agent-assistant/
├── train_lora.py                 # 通用 LoRA 训练脚本
├── train_lora_medical_terms.py   # 医疗术语专项训练脚本
├── lora_inference.py             # 通用推理脚本
├── medical_terms_inference.py    # 医疗术语推理脚本
├── app.py                        # Flask 应用（支持双模式）
├── tools.py                      # 工具模块（集成 LoRA）
├── data/
│   └── medical_finetune.json     # 本地训练数据集
├── lora_output/                  # LoRA 权重输出目录
└── lora_medical_terms/           # 医疗术语模型权重
```

---

## 九、总结

### 9.1 当前状态

| 能力维度 | 状态 | 说明 |
|---------|------|------|
| LoRA 训练 | ✅ 已实现 | 支持 GLM-4-6B/9B |
| 推理部署 | ✅ 已实现 | 本地/API 双模式 |
| 应用集成 | ✅ 已实现 | Flask + 工具系统 |
| 数据准备 | ✅ 已实现 | 本地数据 + 华图数据集 |

### 9.2 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 执行训练
python train_lora_medical_terms.py

# 3. 启动应用
python app.py
```

### 9.3 预期收益

- ✅ 降低 API 调用成本
- ✅ 提升医疗领域问答准确性
- ✅ 数据隐私保护（本地训练）
- ✅ 低延迟响应

---

**项目代码**：https://github.com/AmilyTsang/agent-assistant