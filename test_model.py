# test_model.py
# 测试训练好的模型性能
import os
import json
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from collections import Counter

def load_test_data(test_file_path, num_samples=100):
    """
    从测试文件中随机加载指定数量的样本
    """
    with open(test_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 随机选择样本
    if len(data) > num_samples:
        selected_data = random.sample(data, num_samples)
    else:
        selected_data = data
    
    texts = [item['content'] for item in selected_data]
    labels = [item['label'] for item in selected_data]
    
    print(f"✅ 加载测试数据: {len(selected_data)} 条样本")
    print("测试数据标签分布:", Counter(labels))
    
    return texts, labels

def load_model_and_tokenizer(model_path):
    """
    加载训练好的模型和分词器
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # 加载标签映射
    with open(os.path.join(model_path, 'label_mapping.json'), 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    
    return model, tokenizer, label_mapping

def predict_texts(model, tokenizer, texts, label_mapping, max_length=256):
    """
    对文本进行批量预测
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    predictions = []
    id2label = label_mapping['id2label']
    
    # 加载logit bias（如果存在）
    logit_bias = None
    try:
        # 根据训练时的设置计算logit bias
        label2id = label_mapping['label2id']
        # 这里简化处理，实际应该从训练参数中获取
        logit_bias = torch.zeros(len(label2id))
    except:
        pass
    
    with torch.no_grad():
        for text in texts:
            # 文本预处理（与训练时保持一致）
            import re
            _url = re.compile(r'https?://\S+|www\.\S+')
            _at = re.compile(r'@\S+')
            _topic = re.compile(r'#([^#]+)#')
            _space = re.compile(r'\s+')
            
            clean_text = _url.sub(' ', str(text))
            clean_text = _at.sub(' ', clean_text)
            clean_text = _topic.sub(r'\1', clean_text)
            clean_text = clean_text.replace('转发微博', ' ').replace('来自', ' ')
            clean_text = _space.sub(' ', clean_text).strip()
            
            # 分词和编码
            inputs = tokenizer(clean_text, 
                             return_tensors='pt', 
                             truncation=True, 
                             max_length=max_length,
                             padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 预测
            outputs = model(**inputs)
            logits = outputs.logits
            
            # 应用logit bias（如果有）
            if logit_bias is not None:
                logits = logits + logit_bias.to(device)
            
            pred_id = torch.argmax(logits, dim=-1).item()
            pred_label = id2label[str(pred_id)]
            predictions.append(pred_label)
    
    return predictions

def evaluate_model(true_labels, pred_labels, label_mapping):
    """
    评估模型性能
    """
    # 计算基本指标
    accuracy = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
    
    print(f"\n📊 模型性能评估结果:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"宏平均F1分数: {f1_macro:.4f}")
    print(f"加权平均F1分数: {f1_weighted:.4f}")
    
    # 详细分类报告
    unique_labels = sorted(list(set(true_labels + pred_labels)))
    print(f"\n📋 详细分类报告:")
    print(classification_report(true_labels, pred_labels, 
                              target_names=unique_labels, 
                              digits=4))
    
    # 混淆矩阵
    print(f"\n🔢 混淆矩阵:")
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
    print("标签顺序:", unique_labels)
    print(cm)
    
    # 错误分析
    print(f"\n❌ 错误分析:")
    errors = []
    for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
        if true != pred:
            errors.append((i, true, pred))
    
    print(f"总错误数: {len(errors)} / {len(true_labels)}")
    if len(errors) > 0:
        error_types = Counter([(true, pred) for _, true, pred in errors])
        print("主要错误类型:")
        for (true, pred), count in error_types.most_common(5):
            print(f"  {true} -> {pred}: {count} 次")
    
    return accuracy, f1_macro, f1_weighted

def main():
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 文件路径
    test_file = "./data/test/usual_test_labeled.txt"
    model_path = "./runs/distilbert_balanced20k_retrain"
    
    print("🚀 开始模型测试...")
    
    # 检查文件是否存在
    if not os.path.exists(test_file):
        print(f"❌ 测试文件不存在: {test_file}")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return
    
    # 加载测试数据
    texts, true_labels = load_test_data(test_file, num_samples=1000)
    
    # 加载模型
    print("\n📦 加载训练好的模型...")
    model, tokenizer, label_mapping = load_model_and_tokenizer(model_path)
    print(f"模型标签: {list(label_mapping['label2id'].keys())}")
    
    # 进行预测
    print("\n🔮 开始预测...")
    pred_labels = predict_texts(model, tokenizer, texts, label_mapping)
    
    # 评估性能
    accuracy, f1_macro, f1_weighted = evaluate_model(true_labels, pred_labels, label_mapping)
    
    # 显示一些预测示例
    print(f"\n🔍 预测示例 (前10条):")
    for i in range(min(10, len(texts))):
        status = "✅" if true_labels[i] == pred_labels[i] else "❌"
        print(f"{status} 文本: {texts[i][:50]}...")
        print(f"   真实标签: {true_labels[i]} | 预测标签: {pred_labels[i]}")
        print()
    
    print("\n🎉 测试完成!")

if __name__ == "__main__":
    main()