import onnxruntime as ort
import numpy as np
import sys
from transformers import DistilBertTokenizer

# 检查命令行参数
if len(sys.argv) < 2:
    print("使用方法: python test_distilbert_onnx.py \"您的文本内容\"")
    print("示例: python test_distilbert_onnx.py \"今天天气真好！\"")
    sys.exit(1)

# 从命令行参数获取输入文本
input_text = sys.argv[1]

# ===== 1. 加载 Tokenizer =====
print("📥 加载DistilBERT tokenizer...")
try:
    tokenizer = DistilBertTokenizer.from_pretrained("./distilbert-base-zh-cased")
    print("✅ Tokenizer加载成功")
except Exception as e:
    print(f"❌ Tokenizer加载失败: {e}")
    print("请确保 './distilbert-base-zh-cased' 目录存在")
    sys.exit(1)

# ===== 2. 加载 ONNX 模型 =====
model_path = "distilbert_emotion_analysis.onnx"
print(f"📥 加载ONNX模型: {model_path}...")
try:
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print("✅ ONNX模型加载成功")
except Exception as e:
    print(f"❌ ONNX模型加载失败: {e}")
    print("请先运行 'python convert_distilbert_to_onnx.py' 生成ONNX模型")
    sys.exit(1)

# ===== 3. 定义类别标签 =====
id2label = {0: "负面", 1: "中性", 2: "正面"}
label2emoji = {"负面": "😞", "中性": "😐", "正面": "😊"}
label2color = {"负面": "🔴", "中性": "🟡", "正面": "🟢"}

# ===== 4. Tokenize输入文本 =====
print("🔧 处理输入文本...")
max_length = 128

inputs = tokenizer(
    input_text,
    return_tensors="np",
    padding="max_length",
    truncation=True,
    max_length=max_length
)

print(f"  - 原始文本长度: {len(input_text)} 字符")
print(f"  - Token数量: {np.sum(inputs['attention_mask'])} tokens")
print(f"  - 序列长度: {max_length}")

# ===== 5. ONNX 推理 =====
print("🚀 开始ONNX推理...")
onnx_inputs = {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"]
}

try:
    outputs = session.run(None, onnx_inputs)
    logits = outputs[0]  # shape: [1, 3]
    print("✅ 推理完成")
except Exception as e:
    print(f"❌ 推理失败: {e}")
    sys.exit(1)

# ===== 6. 处理预测结果 =====
print("📊 分析预测结果...")

# 手动实现softmax函数
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 计算概率
probabilities = softmax(logits)[0]  # shape: [3]
predicted_class = int(np.argmax(logits, axis=1)[0])
predicted_label = id2label[predicted_class]
predicted_emoji = label2emoji[predicted_label]
predicted_color = label2color[predicted_label]
confidence = float(probabilities[predicted_class])

# ===== 7. 输出结果 =====
print("\n" + "=" * 70)
print(f"📝 输入文本: {input_text}")
print(f"{predicted_emoji} 预测情感: {predicted_label} {predicted_color}")
print(f"🎯 置信度: {confidence:.4f} ({confidence*100:.2f}%)")
print("=" * 70)

# ===== 8. 显示详细概率分布 =====
print("📊 各类别概率分布:")
for i, (label, prob) in enumerate(zip(["负面", "中性", "正面"], probabilities)):
    emoji = label2emoji[label]
    color = label2color[label]
    
    # 创建可视化进度条
    bar_length = int(prob * 30)  # 30个字符的进度条
    bar = "█" * bar_length + "░" * (30 - bar_length)
    
    # 标记当前预测的类别
    marker = " ← 预测" if i == predicted_class else ""
    
    print(f"  {emoji} {label} {color}: {bar} {prob:.4f} ({prob*100:.2f}%){marker}")

# ===== 9. 置信度评估 =====
print("\n🔍 置信度评估:")
if confidence >= 0.8:
    confidence_level = "非常高"
    confidence_emoji = "🎯"
elif confidence >= 0.6:
    confidence_level = "较高"
    confidence_emoji = "✅"
elif confidence >= 0.4:
    confidence_level = "中等"
    confidence_emoji = "⚠️"
else:
    confidence_level = "较低"
    confidence_emoji = "❓"

print(f"  {confidence_emoji} 置信度等级: {confidence_level}")

# ===== 10. 模型信息 =====
print("\n📋 模型信息:")
print(f"  - 模型类型: DistilBERT中文情感分析")
print(f"  - 输入维度: {inputs['input_ids'].shape}")
print(f"  - 输出维度: {logits.shape}")
print(f"  - 类别数量: {len(id2label)}")
print(f"  - 最大序列长度: {max_length}")

# ===== 11. 性能统计 =====
import time
print("\n⚡ 性能测试 (运行10次推理):")
start_time = time.time()

for _ in range(10):
    session.run(None, onnx_inputs)

end_time = time.time()
avg_time = (end_time - start_time) / 10 * 1000  # 转换为毫秒

print(f"  - 平均推理时间: {avg_time:.2f} ms")
print(f"  - 推理速度: {1000/avg_time:.1f} 次/秒")

print("\n" + "=" * 70)
print("🎉 DistilBERT ONNX情感分析完成！")
print("=" * 70)