import torch
from transformers import BertTokenizer, BertForSequenceClassification
import sys

# 检查命令行参数
if len(sys.argv) < 2:
    print("使用方法: python test_finetuned_model.py \"您的文本内容\"")
    print("示例: python test_finetuned_model.py \"今天天气真好！\"")
    sys.exit(1)

# 从命令行参数获取输入文本
input_text = sys.argv[1]

# ===== 1. 加载微调后的模型和tokenizer =====
model_path = "./tinybert_emotion_finetuned"

try:
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    print("✅ 成功加载微调后的模型")
except Exception as e:
    print(f"❌ 无法加载微调后的模型: {e}")
    print("请先运行 finetune_tinybert.py 进行模型微调")
    sys.exit(1)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# ===== 2. 预处理输入文本 =====
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=128
)

# 将输入移到设备上
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)
token_type_ids = inputs['token_type_ids'].to(device)

# ===== 3. 模型推理 =====
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(logits, dim=1).item()

# ===== 4. 解析结果 =====
id2label = {0: "负面", 1: "中性", 2: "正面"}
label2emoji = {"负面": "😞", "中性": "😐", "正面": "😊"}

predicted_label = id2label[predicted_class]
predicted_emoji = label2emoji[predicted_label]

# ===== 5. 输出结果 =====
print("=" * 60)
print(f"📝 输入文本: {input_text}")
print(f"{predicted_emoji} 预测情感: {predicted_label}")
print(f"🎯 置信度: {probabilities[0][predicted_class].item():.4f}")
print("=" * 60)

# 显示所有类别的概率
print("📊 各类别概率分布:")
for i, (label, prob) in enumerate(zip(["负面", "中性", "正面"], probabilities[0])):
    emoji = label2emoji[label]
    bar_length = int(prob.item() * 20)  # 20个字符的进度条
    bar = "█" * bar_length + "░" * (20 - bar_length)
    print(f"  {emoji} {label}: {bar} {prob.item():.4f}")

print("=" * 60)