import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import os

# ===== 1. 数据集类定义 =====
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ===== 2. 创建示例数据集 =====
def create_sample_dataset():
    """创建一个示例情感分析数据集"""
    # 示例数据 - 实际使用时应该用真实的大规模数据集
    texts = [
        "今天天气真好，心情特别棒！",
        "这部电影太糟糕了，完全浪费时间",
        "还可以吧，没什么特别的",
        "非常喜欢这个产品，质量很好",
        "服务态度很差，很失望",
        "一般般，没有惊喜也没有失望",
        "太棒了！超出预期",
        "质量有问题，不推荐购买",
        "普通的产品，价格合理",
        "excellent quality, highly recommend",
        "terrible experience, waste of money",
        "average product, nothing special",
        "amazing service, very satisfied",
        "poor quality, disappointed",
        "okay product, fair price",
        "love it so much, perfect",
        "worst purchase ever made",
        "decent quality for the price",
        "outstanding performance, impressed",
        "not worth the money, regret buying"
    ]
    
    # 标签: 0=负面, 1=中性, 2=正面
    labels = [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    
    return texts, labels

# ===== 3. 训练函数 =====
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions
    
    return avg_loss, accuracy

# ===== 4. 评估函数 =====
def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions
    
    return avg_loss, accuracy, all_predictions, all_labels

# ===== 5. 主训练流程 =====
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集
    texts, labels = create_sample_dataset()
    
    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"训练集大小: {len(train_texts)}")
    print(f"验证集大小: {len(val_texts)}")
    
    # 加载tokenizer和模型
    model_name = "huawei-noah/TinyBERT_General_4L_312D"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # 创建分类模型（3个类别：负面、中性、正面）
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label={0: "负面", 1: "中性", 2: "正面"},
        label2id={"负面": 0, "中性": 1, "正面": 2}
    )
    model.to(device)
    
    # 创建数据加载器
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # 设置优化器和学习率调度器
    epochs = 3
    total_steps = len(train_loader) * epochs
    
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_accuracy = 0
    
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        
        # 验证
        val_loss, val_acc, val_predictions, val_labels = evaluate_model(model, val_loader, device)
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            model.save_pretrained("./TinyBERT/tinybert_emotion_finetuned")
            tokenizer.save_pretrained("./TinyBERT/tinybert_emotion_finetuned")
            print(f"保存最佳模型，验证准确率: {best_accuracy:.4f}")
    
    # 最终评估报告
    print("\n=== 最终评估报告 ===")
    print(classification_report(
        val_labels, 
        val_predictions, 
        target_names=["负面", "中性", "正面"]
    ))
    
    print(f"\n微调完成！最佳验证准确率: {best_accuracy:.4f}")
    print("微调后的模型已保存到 './TinyBERT/tinybert_emotion_finetuned' 目录")

if __name__ == "__main__":
    main()