# part1_train_weighted.py
import os, argparse, json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import torch
from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer, set_seed)

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        if self.class_weights is not None:
            loss = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))(outputs.logits, labels)
        else:
            loss = torch.nn.CrossEntropyLoss()(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss
def load_or_make_dataset(csv_path, seed, limit_train=None, limit_val=None):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from collections import Counter

    if csv_path and os.path.exists(csv_path):
        # 1) 自动探测分隔符读取
        df = pd.read_csv(csv_path, sep=None, engine="python", dtype=str)
        df = df.dropna(how="all")
        # 2) 自动定位标签列：数值且子集在 {0,1,2,3}
        label_col = None
        for c in df.columns:
            col = pd.to_numeric(df[c], errors="coerce").dropna().astype(int)
            ok = set(col.unique())
            if ok.issubset({0,1,2,3}) and len(ok) >= 3:
                label_col = c
                break
        if label_col is None:
            raise ValueError("未找到标签列（0/1/2/3）。请检查文件分隔符或提供列名。")

        # 3) 规范化标签为 int，其他列拼接为 text
        df["label"] = pd.to_numeric(df[label_col], errors="coerce").astype("Int64")
        text_cols = [c for c in df.columns if c != label_col]
        df["text"] = df[text_cols].astype(str).apply(lambda r: " ".join([x for x in r if x and x != "nan"]), axis=1)

        # 4) 只保留合法四类 + 文本非空
        df = df[df["label"].isin([0,1,2,3]) & df["text"].str.strip().ne("")]
        df = df[["label", "text"]].dropna()
        # 5) 映射到四个情绪名
        label_map = {0: "joy", 1: "anger", 2: "disgust", 3: "depress"}
        df["label"] = df["label"].map(label_map)

        print(f"✅ 加载CSV数据: {len(df)} 条样本")
        print("整体标签分布:", Counter(df["label"].astype(str)))
    else:
        # 回退到内置小样本
        texts = [
            "今天心情很好，效率也很高！",
            "真是气死我了，服务太差。",
            "一般般吧，没什么感觉。",
            "这个功能做得不错，体验很棒！",
            "糟透了，再也不会用了。",
            "还行吧，中规中矩。"
        ]
        labels = ["joy", "anger", "disgust", "joy", "anger", "disgust"]
        df = pd.DataFrame({"text": texts, "label": labels})
        print("ℹ️ 未提供 CSV，使用内置小样例数据。")

    # —— 确保四类都存在（很关键）——
    present = set(df["label"].unique().tolist())
    need = {"joy","anger","disgust","depress"}
    if not need.issubset(present):
        print("⚠️ 警告：数据集中缺少类别：", need - present, "请检查解析是否正确。")

    # 构建映射
    uniq = sorted(df["label"].astype(str).unique().tolist())
    label2id = {lab: i for i, lab in enumerate(uniq)}
    id2label = {i: lab for lab, i in label2id.items()}

    # 先 8:2 切分，再按需限量且保持分布
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["label"].astype(str)
    )
    if limit_train and len(train_df) > limit_train:
        train_df, _ = train_test_split(
            train_df, train_size=limit_train, random_state=seed,
            stratify=train_df["label"].astype(str)
        )
    if limit_val and len(val_df) > limit_val:
        val_df, _ = train_test_split(
            val_df, train_size=limit_val, random_state=seed,
            stratify=val_df["label"].astype(str)
        )

    print(f"train size: {len(train_df)}, val size: {len(val_df)}")
    print("train label dist:", Counter(train_df["label"].astype(str)))
    print("val   label dist:", Counter(val_df["label"].astype(str)))

    # 数值 id
    train_df["label"] = train_df["label"].astype(str).map(label2id)
    val_df["label"]   = val_df["label"].astype(str).map(label2id)

    from datasets import Dataset, DatasetDict
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
    ap.add_argument("--out_dir", type=str, default="./runs/distilbert_zh_5k_weighted")
    ap.add_argument("--epochs_head", type=float, default=1.0, help="只训分类头的轮数")
    ap.add_argument("--epochs_full", type=float, default=2.0, help="解冻全模型后的轮数")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_train", type=int, default=5000)
    ap.add_argument("--limit_val", type=int, default=1000)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    model_name = "distilbert-base-zh-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("[Tokenize 检查]", tokenizer.tokenize("今天心情很好，效率也很高！")[:20])

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
    model.config.problem_type = "single_label_classification"

    # 类别权重：N / (K * n_i)
    cnt = Counter(ds["train"]["label"])
    K = len(cnt)
    N = len(ds["train"])
    class_weights = torch.tensor([N / (K * cnt[i]) for i in range(K)], dtype=torch.float)

    data_collator = DataCollatorWithPadding(tokenizer)

    # -------- 阶段1：只训分类头 --------
    for n, p in model.named_parameters():
        p.requires_grad = ("classifier" in n)

    args_head = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "stage1_head"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,          # 头部大一点
        warmup_ratio=0.0,            # 不预热
        weight_decay=0.0,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size*2),
        num_train_epochs=args.epochs_head,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=1
    )

    trainer1 = WeightedTrainer(
        class_weights=class_weights,
        model=model, args=args_head,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        tokenizer=tokenizer, data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer1.train()
    print("✅ 阶段1完成")

    # -------- 阶段2：解冻全模型，细调 --------
    for p in model.parameters():
        p.requires_grad = True

    args_full = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "stage2_full"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,          # 全量微调用小LR
        warmup_ratio=0.1,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size*2),
        num_train_epochs=args.epochs_full,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=1
    )

    trainer2 = WeightedTrainer(
        class_weights=class_weights,
        model=model, args=args_full,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        tokenizer=tokenizer, data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer2.train()
    metrics = trainer2.evaluate()
    print("✅ 最终评估:", metrics)

    # 保存最终模型
    trainer2.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
    print(f"✅ 保存到 {args.out_dir}")

    # ------ 训练后报告 ------
    preds = trainer2.predict(ds["validation"])
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=-1)
    print("\n[分类报告]\n", classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(num_labels)], digits=4))
    print("[混淆矩阵]\n", confusion_matrix(y_true, y_pred))

    # ------ 快速自测 ------
    model.eval()
    device = next(model.parameters()).device  # 获取模型所在设备
    samples = ["今天心情很好，效率也很高！", "真是气死我了，服务太差。", "一般般吧，没什么感觉。"]
    for s in samples:
        enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=args.max_len)
        # 将输入数据移动到模型所在设备
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            pred = int(model(**enc).logits.argmax(dim=-1).item())
        print(f"[自测] {s} -> {id2label[pred]}")

if __name__ == "__main__":
    main()
