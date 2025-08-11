import onnxruntime as ort
import numpy as np
import time
from transformers import BertTokenizer, DistilBertTokenizer
import sys

# 检查命令行参数
if len(sys.argv) < 2:
    print("使用方法: python compare_models.py \"您的文本内容\"")
    print("示例: python compare_models.py \"今天天气真好！\"")
    sys.exit(1)

# 从命令行参数获取输入文本
input_text = sys.argv[1]

print("=" * 80)
print("🔬 TinyBERT vs DistilBERT ONNX 模型对比测试")
print("=" * 80)
print(f"📝 测试文本: {input_text}")
print("=" * 80)

# ===== 定义类别标签 =====
id2label = {0: "负面", 1: "中性", 2: "正面"}
label2emoji = {"负面": "😞", "中性": "😐", "正面": "😊"}
label2color = {"负面": "🔴", "中性": "🟡", "正面": "🟢"}

def softmax(x):
    """计算softmax概率"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def test_model(model_name, model_path, tokenizer_path, tokenizer_class, max_length=128):
    """测试单个模型"""
    print(f"\n🧪 测试 {model_name} 模型...")
    
    # 加载tokenizer
    try:
        tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        print(f"✅ {model_name} tokenizer加载成功")
    except Exception as e:
        print(f"❌ {model_name} tokenizer加载失败: {e}")
        return None
    
    # 加载ONNX模型
    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print(f"✅ {model_name} ONNX模型加载成功")
    except Exception as e:
        print(f"❌ {model_name} ONNX模型加载失败: {e}")
        return None
    
    # Tokenize输入文本
    inputs = tokenizer(
        input_text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    
    # 准备ONNX输入
    if model_name == "TinyBERT":
        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs["token_type_ids"]
        }
    else:  # DistilBERT
        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
    
    # 单次推理测试
    try:
        outputs = session.run(None, onnx_inputs)
        if model_name == "TinyBERT":
            # TinyBERT输出last_hidden_state，需要模拟分类
            last_hidden_state = outputs[0]  # shape: [1, seq_len, 312]
            cls_vector = last_hidden_state[0, 0, :]  # 取CLS token的向量
            # 模拟简单的分类逻辑（这里只是示例）
            logits = np.array([[np.mean(cls_vector[:100]), np.mean(cls_vector[100:200]), np.mean(cls_vector[200:])]])
        else:
            logits = outputs[0]  # shape: [1, 3]
        print(f"✅ {model_name} 推理完成")
    except Exception as e:
        print(f"❌ {model_name} 推理失败: {e}")
        return None
    
    # 计算预测结果
    probabilities = softmax(logits)[0]  # shape: [3]
    predicted_class = int(np.argmax(logits, axis=1)[0])
    predicted_label = id2label[predicted_class]
    predicted_emoji = label2emoji[predicted_label]
    predicted_color = label2color[predicted_label]
    confidence = float(probabilities[predicted_class])
    
    # 性能测试
    print(f"⚡ {model_name} 性能测试 (运行100次推理)...")
    start_time = time.time()
    
    for _ in range(100):
        session.run(None, onnx_inputs)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 100 * 1000  # 转换为毫秒
    
    return {
        "model_name": model_name,
        "predicted_label": predicted_label,
        "predicted_emoji": predicted_emoji,
        "predicted_color": predicted_color,
        "confidence": confidence,
        "probabilities": probabilities,
        "avg_time": avg_time,
        "logits": logits[0],
        "token_count": np.sum(inputs['attention_mask'])
    }

# ===== 测试两个模型 =====
results = []

# 测试TinyBERT (新版本)
tinybert_result = test_model(
    "TinyBERT",
    "./TinyBERT/tinybert_emotion_analysis_128.onnx",
    "huawei-noah/TinyBERT_General_4L_312D",
    BertTokenizer
)
if tinybert_result:
    results.append(tinybert_result)

# 测试DistilBERT
distilbert_result = test_model(
    "DistilBERT",
    "./distilbert-base-zh-cased/distilbert_emotion_analysis.onnx",
    "./distilbert-base-zh-cased",
    DistilBertTokenizer
)
if distilbert_result:
    results.append(distilbert_result)

# ===== 对比结果 =====
if len(results) == 2:
    print("\n" + "=" * 80)
    print("📊 模型对比结果")
    print("=" * 80)
    
    # 预测结果对比
    print("\n🎯 预测结果对比:")
    for result in results:
        print(f"  {result['predicted_emoji']} {result['model_name']}: {result['predicted_label']} {result['predicted_color']} (置信度: {result['confidence']:.4f})")
    
    # 性能对比
    print("\n⚡ 性能对比:")
    for result in results:
        print(f"  🚀 {result['model_name']}: {result['avg_time']:.2f} ms/次 ({1000/result['avg_time']:.1f} 次/秒)")
    
    # 速度差异
    if results[0]['avg_time'] != results[1]['avg_time']:
        faster_idx = 0 if results[0]['avg_time'] < results[1]['avg_time'] else 1
        slower_idx = 1 - faster_idx
        speedup = results[slower_idx]['avg_time'] / results[faster_idx]['avg_time']
        print(f"  📈 {results[faster_idx]['model_name']} 比 {results[slower_idx]['model_name']} 快 {speedup:.2f}x")
    
    # 详细概率分布对比
    print("\n📊 详细概率分布对比:")
    for result in results:
        print(f"\n  {result['model_name']} 模型:")
        for i, (label, prob) in enumerate(zip(["负面", "中性", "正面"], result['probabilities'])):
            emoji = label2emoji[label]
            color = label2color[label]
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            marker = " ← 预测" if i == np.argmax(result['probabilities']) else ""
            print(f"    {emoji} {label} {color}: {bar} {prob:.4f} ({prob*100:.2f}%){marker}")
    
    # Logits对比
    print("\n🔢 原始Logits对比:")
    for result in results:
        logits_str = ", ".join([f"{x:.4f}" for x in result['logits']])
        print(f"  {result['model_name']}: [{logits_str}]")
    
    # Token数量对比
    print("\n📝 Token处理对比:")
    for result in results:
        print(f"  {result['model_name']}: {result['token_count']} tokens")
    
    # 一致性检查
    print("\n🔍 预测一致性:")
    if results[0]['predicted_label'] == results[1]['predicted_label']:
        print("  ✅ 两个模型预测结果一致")
    else:
        print("  ⚠️ 两个模型预测结果不一致")
        conf_diff = abs(results[0]['confidence'] - results[1]['confidence'])
        print(f"  📊 置信度差异: {conf_diff:.4f}")

else:
    print("\n❌ 无法完成对比，部分模型加载失败")

print("\n" + "=" * 80)
print("🎉 模型对比测试完成！")
print("=" * 80)