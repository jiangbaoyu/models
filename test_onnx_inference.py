import onnxruntime as ort
import numpy as np
from transformers import BertTokenizer

# ===== 1. 加载 Tokenizer =====
tokenizer = BertTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

# ===== 2. 加载 ONNX 模型 =====
model_path = "tinybert_emotion_analysis_128.onnx"
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# ===== 3. 定义类别标签 =====
id2label = {0: "负面", 1: "中性", 2: "正面"}

# ===== 4. 准备输入文本 =====
text = "雨夜，独行在街道上,很冷，发抖。"

# ===== 5. Tokenize（保证最大长度 ≤ 128） =====
inputs = tokenizer(
    text,
    return_tensors="np",   # 直接输出 NumPy
    padding="max_length",
    truncation=True,
    max_length=128
)

# ===== 6. ONNX 推理 =====
onnx_inputs = {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
    "token_type_ids": inputs["token_type_ids"]
}

outputs = session.run(None, onnx_inputs)

# ===== 7. 获取第二个输出 (CLS 向量) =====
cls_vector = outputs[1]  # shape: [1, 312]

# ===== 8. 模拟分类权重（这里是示例，实际应使用训练好的分类头） =====
# 假设分类层权重 W: [312, 3], 偏置 b: [3]
# 这里随机初始化一个示例
W = np.random.randn(312, 3)
b = np.random.randn(3)

# 计算 logits
logits = np.dot(cls_vector, W) + b
pred_label_id = int(np.argmax(logits, axis=1)[0])
pred_label = id2label[pred_label_id]

print(f"输入文本: {text}")
print(f"预测情感: {pred_label}")
print(f"CLS 向量形状: {cls_vector.shape}")
print(f"模型输出数量: {len(outputs)}")
print(f"第一个输出形状: {outputs[0].shape}")
print(f"第二个输出形状: {outputs[1].shape}")