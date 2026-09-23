# Demo 3：Qwen2.5-7B 指令微调实体识别

本项目基于 Qwen2.5-7B 模型，在 BC2GM 生物医学数据集上完成生成式命名实体识别（NER）任务，并使用指令微调方式训练模型识别句子中的 GENE 实体。

```
安装示例：

```bash
python -m pip install -U \
  torch transformers peft accelerate bitsandbytes \
  liger-kernel swanlab datasets tqdm safetensors
```

## 项目结构

```text
demo3/
├── config/
│   ├── lora_config.json          # BF16 LoRA 配置
│   └── qlora_config.json         # 4-bit QLoRA 配置
├── data/
│   ├── train.json                # BC2GM 原始训练数据
│   ├── dev.json                  # BC2GM 原始验证数据
│   ├── test.json                 # BC2GM 原始测试数据
│   ├── bc_train.json             # 指令格式训练数据
│   ├── bc_dev.json               # 指令格式验证数据
│   ├── bc_test.json              # 指令格式测试数据
│   └── data_preprocess.py        # 数据预处理脚本
├── picture                       # 存放运行图片
├── pretrain/
│   └── Qwen_7B/                  # Qwen2.5-7B 本地模型目录
├── config.py                     # 配置加载
├── dataset.py                    # 数据处理、ChatML 和预 tokenize
├── model_m.py                    # 模型加载、量化和 LoRA 注入
├── trainer.py                    # 训练、验证、测试和日志
├── train.py                      # 训练入口
├── test.py                       # 测试入口
├── requriments                   # 需求文档
└── metrics.py                    # 实体级 P/R/F1

```

## 数据说明

BC2GM 数据规模：

```text
训练集: 12,500
验证集: 2,500
测试集: 5,000
```

指令数据使用以下输出格式：

```text
BRCA1:GENE
TP53:GENE
```

没有实体时输出：

```text
无实体
```

数据预处理命令：

```bash
cd data
python data_preprocess.py
cd ..
```

预处理后会生成：

```text
bc_train.json
bc_dev.json
bc_test.json
```

## 指令与训练格式

Dataset 使用 Qwen ChatML：

```text
<|im_start|>system
You are a biomedical NER system. Extract all GENE entities...
<|im_end|>
<|im_start|>user
待识别句子
<|im_end|>
<|im_start|>assistant
BRCA1:GENE
<|im_end|>
```

训练时只对 assistant 回答部分计算 loss，system 和 user 部分的 labels
设置为 `-100`。数据在 Dataset 初始化阶段预 tokenize，避免每个 epoch 重复分词。

## 模型与微调配置

LoRA 与 QLoRA 使用相同的 LoRA 结构：

```text
lora_rank = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

LoRA：

```text
基座精度: BF16
量化: 无
gradient_checkpointing: false
batch_size: 4
gradient_accumulation_steps: 2
```

QLoRA：

```text
基座精度: 4-bit NF4
bnb_4bit_compute_dtype: bfloat16
双量化: true
gradient_checkpointing: true
batch_size: 4
gradient_accumulation_steps: 2
```

公共训练设置：

```text
epochs: 3
learning_rate: 5e-5
lr_scheduler: cosine
weight_decay: 0.01
max_grad_norm: 1.0
seed: 42
```

## 训练

QLoRA：

```bash
python train.py --config config/qlora_config.json
```

LoRA：

```bash
python train.py --config config/lora_config.json
```

检查点保存位置：

```text
./ckpt/7b_qlora/epoch_1
./ckpt/7b_qlora/epoch_2
./ckpt/7b_qlora/epoch_3

./ckpt/7b_lora/epoch_1
./ckpt/7b_lora/epoch_2
./ckpt/7b_lora/epoch_3
```

每个 epoch 会记录 `train_loss` 和 `dev_loss`。训练全部结束后，会再对完整
dev 集计算一次实体级 P/R/F1。

## 测试

QLoRA：

```bash
python test.py --config config/qlora_config.json
```

LoRA：

```bash
python test.py --config config/lora_config.json
```

### 总体结果
#### Lora未开启梯度检查，Qlora开启梯度检查

| 方法 | Precision | Recall |     F1 | 训练峰值显存 |
| --- |----------:|-------:|-------:|-------:|
| QLoRA |    0.8352 | 0.8356 | 0.8354 |   10.4 |
| LoRA |    0.8389 | 0.8282 | 0.8335 |   28.2 |


