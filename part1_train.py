# part1_train.py  (支持 --limit_train / --limit_val)
import os, argparse, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

import torch
from datasets import Dataset, DatasetDict
import evaluate
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer, set_seed)

def load_or_make_dataset(csv_path, seed, limit_train=None, limit_val=None):
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, header=None, names=["label", "text"])
        # 清理数据：删除缺失值
        df = df.dropna()
        # 将数字标签映射为文本标签
        label_map = {0: "joy", 1: "anger", 2: "disgust", 3: "depress"}
        df["label"] = df["label"].map(label_map)
        # 删除映射失败的行（标签不在0-3范围内）
        df = df.dropna()
        print(f"✅ 加载CSV数据: {len(df)} 条样本")
        print(f"标签分布: {df['label'].value_counts().to_dict()}")
    else:
        texts = [
            "今天心情很好，效率也很高！",
            "真是气死我了，服务太差。",
            "一般般吧，没什么感觉。",
            "这个功能做得不错，体验很棒！",
            "糟透了，再也不会用了。",
            "还行吧，中规中矩。"
        ]
        labels = ["positive", "negative", "neutral", "positive", "negative", "neutral"]
        df = pd.DataFrame({"text": texts, "label": labels})
        print("ℹ️ 未提供 CSV，使用内置小样例数据（仅用于流程验证）。")

    # 统一为字符串标签，构建映射
    uniq = sorted(df["label"].astype(str).unique().tolist())
    label2id = {lab: i for i, lab in enumerate(uniq)}
    id2label = {i: lab for lab, i in label2id.items()}

    # 先切分，再对训练/验证分别做限量抽样（保持标签分布）
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["label"].astype(str)
    )

    if limit_train and len(train_df) > limit_train:
        train_df, _ = train_test_split(
            train_df,
            train_size=limit_train,
            random_state=seed,
            stratify=train_df["label"].astype(str)
        )
    if limit_val and len(val_df) > limit_val:
        val_df, _ = train_test_split(
            val_df,
            train_size=limit_val,
            random_state=seed,
            stratify=val_df["label"].astype(str)
        )

    # 打印抽样后标签分布
    print(f"train size: {len(train_df)}, val size: {len(val_df)}")
    print("train label dist:", Counter(train_df["label"].astype(str)))
    print("val   label dist:", Counter(val_df["label"].astype(str)))

    train_df["label"] = train_df["label"].astype(str).map(label2id)
    val_df["label"]    = val_df["label"].astype(str).map(label2id)

    ds = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
    })
    return ds, label2id, id2label

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro")
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="./runs/distilbert_zh")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_train", type=int, default=5000, help="训练集最大样本数（按标签分布抽样）")
    ap.add_argument("--limit_val", type=int, default=1000, help="验证集最大样本数（可选）")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    model_name = "distilbert-base-zh-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sample_tokens = tokenizer.tokenize("今天心情很好，效率也很高！")
    print("[Tokenize 检查]", sample_tokens[:20])

    ds, label2id, id2label = load_or_make_dataset(
        args.csv, args.seed, limit_train=args.limit_train, limit_val=args.limit_val
    )

    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)
    keep_cols = ["input_ids", "attention_mask", "label"]
    ds = ds.map(preprocess, batched=True,
                remove_columns=[c for c in ds["train"].column_names if c not in keep_cols])

    num_labels = len(label2id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size*2),
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print("✅ Eval:", metrics)

    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
    print(f"✅ 模型与分词器保存到 {args.out_dir}")

    # ====== 训练后快速自测 ======
    model.eval()
    device = next(model.parameters()).device  # 获取模型所在设备
    test_texts = ["今天心情很好，效率也很高！", "真是气死我了，服务太差。", "一般般吧，没什么感觉。"]
    for s in test_texts:
        enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=args.max_len)
        # 将输入数据移动到模型所在设备
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            pred = int(logits.argmax(dim=-1).item())
        print(f"[训练后自测] {s} -> {id2label[pred]}")

if __name__ == "__main__":
    main()
