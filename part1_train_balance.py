# part1_train_balanced.py
# 稳健数据加载 + 均衡采样 + Focal Loss（阶段1）+ Logit-Adjusted CE（阶段2）+ 两阶段训练
# 兼容旧版 transformers：make_training_args() 自动适配 evaluation/save 策略 & 关闭不支持的功能
import os, argparse, json, re, inspect
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer, set_seed
)

# =========================
# 兼容老/新版本 TrainingArguments 的构造器
# =========================
def make_training_args(**kw):
    """
    - 新版(支持 evaluation_strategy/save_strategy)：按 epoch 评估&保存，支持 load_best_model_at_end
    - 老版：移除不支持参数，退回 evaluate_during_training + save_steps/eval_steps，
            并关闭 load_best_model_at_end，避免策略不匹配报错
    """
    sig = inspect.signature(TrainingArguments.__init__)
    params = set(sig.parameters.keys())

    # 想要的新API默认
    desired_defaults = dict(
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )
    for k, v in desired_defaults.items():
        kw.setdefault(k, v)

    new_api = ("evaluation_strategy" in params) and ("save_strategy" in params)

    if not new_api:
        # 老API：删掉不存在的键
        kw.pop("evaluation_strategy", None)
        kw.pop("save_strategy", None)
        for k in ["report_to", "metric_for_best_model", "greater_is_better",
                  "warmup_ratio", "save_total_limit", "max_grad_norm"]:
            if k not in params:
                kw.pop(k, None)

        # 开启老式评估
        if "evaluate_during_training" in params:
            kw["evaluate_during_training"] = True
        # 对齐 steps 策略
        if "save_steps" in params and "save_steps" not in kw:
            kw["save_steps"] = 500
        if "eval_steps" in params and "eval_steps" not in kw:
            kw["eval_steps"] = kw.get("save_steps", 500)

        # 关键：避免旧版在 __post_init__ 校验 load_best_model_at_end 时抛错
        if "load_best_model_at_end" in params:
            kw["load_best_model_at_end"] = False

    # 过滤签名外参数
    filtered = {k: v for k, v in kw.items() if k in params}
    return TrainingArguments(**filtered)

# =========================
# 强化版 CSV 读取与清洗
# =========================
def load_clean_csv(csv_path: str) -> pd.DataFrame:
    """
    - 自动探测分隔符（sep=None, engine='python'）
    - 自动识别标签列（值应在 {0,1,2,3}，且至少见到 3 类）
    - 其余列拼接为 text
    - 文本清洗：去 URL/@/#话题#、多空白；过滤空文本；去重
    """
    df = pd.read_csv(csv_path, sep=None, engine="python", header=None, keep_default_na=False)
    df = df.replace(r"^\s*$", np.nan, regex=True).dropna(how="all").reset_index(drop=True)

    # 自动识别标签列
    candidates = []
    for c in df.columns:
        col_num = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")
        valid = col_num.dropna().astype(int)
        uniq = set(valid.unique().tolist())
        if uniq.issubset({0,1,2,3}) and len(uniq) >= 3:
            score = len(valid) / len(df)
            candidates.append((c, score, len(df) - len(valid)))
    if not candidates:
        raise ValueError("未能自动识别标签列（期望包含 0/1/2/3 的整数标签）。请检查数据文件。")
    candidates.sort(key=lambda x: (-x[1], x[2]))
    label_col = candidates[0][0]

    labels = pd.to_numeric(df[label_col].astype(str).str.strip(), errors="coerce").astype("Int64")
    text_cols = [c for c in df.columns if c != label_col]
    if not text_cols:
        raise ValueError("未找到文本列。请确认数据格式至少包含 label + text 两列。")

    # 拼接为 text
    text = (
        df[text_cols].astype(str)
        .apply(lambda r: " ".join([x for x in r if x and x.lower() != "nan"]), axis=1)
    )

    out = pd.DataFrame({"label": labels, "text": text})

    # 文本清洗
    _url = re.compile(r'https?://\S+|www\.\S+'); _at = re.compile(r'@\S+')
    _topic = re.compile(r'#([^#]+)#'); _space = re.compile(r'\s+')
    def clean_text(s: str) -> str:
        s = _url.sub(' ', str(s)); s = _at.sub(' ', s); s = _topic.sub(r'\1', s)
        s = s.replace('转发微博',' ').replace('来自',' '); s = _space.sub(' ', s).strip()
        return s

    out["text"] = out["text"].astype(str).apply(clean_text)
    out = out[out["label"].isin([0,1,2,3]) & out["text"].str.len().ge(2)]
    out = out.drop_duplicates(subset=["label","text"]).reset_index(drop=True)
    return out

# =========================
# 构建 Datasets
# =========================
def load_or_make_dataset(csv_path, seed, limit_train=None, limit_val=None):
    if csv_path and os.path.exists(csv_path):
        df = load_clean_csv(csv_path)
        label_map = {0:"joy", 1:"anger", 2:"disgust", 3:"depress"}
        df["label"] = df["label"].map(label_map)
        print(f"✅ 加载CSV数据: {len(df)} 条样本")
        print("整体标签分布:", Counter(df["label"]))
    else:
        texts  = ["今天心情很好，效率也很高！","真是气死我了，服务太差。","一般般吧，没什么感觉。","这个功能做得不错，体验很棒！","糟透了，再也不会用了。","还行吧，中规中矩。"]
        labels = ["joy","anger","disgust","joy","anger","disgust"]
        df = pd.DataFrame({"text":texts,"label":labels})
        print("ℹ️ 未提供 CSV，使用内置小样例数据。")

    uniq = sorted(df["label"].astype(str).unique().tolist())
    label2id = {lab:i for i, lab in enumerate(uniq)}
    id2label = {i:lab for lab,i in label2id.items()}

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["label"].astype(str))
    if limit_train and len(train_df) > limit_train:
        train_df, _ = train_test_split(train_df, train_size=limit_train, random_state=seed, stratify=train_df["label"].astype(str))
    if limit_val and len(val_df) > limit_val:
        val_df, _ = train_test_split(val_df, train_size=limit_val, random_state=seed, stratify=val_df["label"].astype(str))

    print(f"train size: {len(train_df)}, val size: {len(val_df)}")
    print("train label dist:", Counter(train_df["label"].astype(str)))
    print("val   label dist:", Counter(val_df["label"].astype(str)))

    train_df["label"] = train_df["label"].astype(str).map(label2id)
    val_df["label"]   = val_df["label"].astype(str).map(label2id)

    ds = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
    })
    return ds, label2id, id2label

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro")}

# 加偏指标（阶段2使用）
def make_compute_metrics_with_bias(bias_vec: np.ndarray):
    def _cm(eval_pred):
        logits, labels = eval_pred
        logits = logits + bias_vec[None, :]
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro")
        }
    return _cm

# =========================
# Focal Loss（阶段1）+ 均衡采样 Trainer
# =========================
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = None if alpha is None else torch.tensor(alpha, dtype=torch.float32)

    def forward(self, logits, target):
        ce = torch.nn.functional.cross_entropy(logits, target, reduction="none")
        pt = torch.softmax(logits, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1).clamp_(1e-6, 1.0)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, target)
            loss = alpha_t * loss
        return loss.mean() if self.reduction == "mean" else loss.sum()

class BalancedTrainer(Trainer):
    """阶段1：均衡采样 + FocalLoss"""
    def __init__(self, train_labels=None, focal_alpha=None, focal_gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_labels = train_labels
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    # 兼容不同版本 Trainer 的调用签名
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.focal(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        if self.train_labels is None:
            return super().get_train_dataloader()
        counts = Counter(self.train_labels)
        w_per_class = {c: 1.0 / max(1, n) for c, n in counts.items()}
        sample_weights = [w_per_class[int(y)] for y in self.train_labels]
        sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.double),
                                        num_samples=len(sample_weights),
                                        replacement=True)
        num_workers = getattr(self.args, "dataloader_num_workers", 0)
        return DataLoader(self.train_dataset,
                          batch_size=self.args.train_batch_size,
                          sampler=sampler,
                          collate_fn=self.data_collator,
                          num_workers=num_workers)

# =========================
# 阶段2：标准采样 + Logit-Adjusted CE Trainer
# =========================
class LogitAdjustedCETrainer(Trainer):
    def __init__(self, logit_bias=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # logit_bias: shape [num_labels]，= -tau * log(p_y)
        self.logit_bias = torch.tensor(logit_bias, dtype=torch.float32) if logit_bias is not None else None

    # 兼容不同版本 Trainer 的调用签名
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.logit_bias is not None:
            logits = logits + self.logit_bias.to(logits.device)  # 训练时先验校准
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss

# =========================
# 主函数
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="simplifyweibo_4_moods.csv")
    ap.add_argument("--out_dir", type=str, default="./runs/distilbert_balanced")
    ap.add_argument("--epochs_head", type=float, default=1.0)
    ap.add_argument("--epochs_full", type=float, default=6.0)   # 稍多一轮
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_train", type=int, default=10000)   # 可改 20000 更稳
    ap.add_argument("--limit_val", type=int, default=4000)
    ap.add_argument("--tau", type=float, default=1.0, help="Logit-adjustment 的温度（建议 0.7~1.2）")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    model_name = "distilbert-base-zh-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("[Tokenize 检查]", tokenizer.tokenize("今天心情很好，效率也很高！")[:20])

    # 数据集
    ds, label2id, id2label = load_or_make_dataset(args.csv, args.seed, args.limit_train, args.limit_val)

    # 训练时轻清洗 + tokenize
    _url = re.compile(r'https?://\S+|www\.\S+'); _at = re.compile(r'@\S+')
    _topic = re.compile(r'#([^#]+)#'); _space = re.compile(r'\s+')
    def clean_text(s: str) -> str:
        s = _url.sub(' ', str(s)); s = _at.sub(' ', s); s = _topic.sub(r'\1', s)
        s = s.replace('转发微博',' ').replace('来自',' '); s = _space.sub(' ', s).strip()
        return s
    def preprocess(b):
        texts = [clean_text(t) for t in b["text"]]
        return tokenizer(texts, truncation=True, max_length=args.max_len)
    keep = ["input_ids","attention_mask","label"]
    ds = ds.map(preprocess, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in keep])

    # 模型
    num_labels = len(label2id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    )
    model.config.problem_type = "single_label_classification"
    collator = DataCollatorWithPadding(tokenizer)

    # 统计分布（用于 α 与 logit_bias）
    train_labels = list(ds["train"]["label"])
    cnt = Counter(train_labels); K = len(cnt); N = len(train_labels)

    # FocalLoss 的 alpha（逆频率，均值归一到 1）
    alpha = np.array([N/(K*cnt[i]) for i in range(K)], dtype=np.float32)
    alpha = alpha * (K / alpha.sum())

    # Logit-Adjusted CE 的偏置：-tau * log(p_y)
    p = np.array([cnt[i] / N for i in range(K)], dtype=np.float32)
    tau = float(args.tau)
    logit_bias = (-tau) * np.log(np.clip(p, 1e-12, 1.0))  # [K]

    # ========= 阶段1：只训分类头（均衡采样 + Focal）=========
    for n, p_ in model.named_parameters():
        p_.requires_grad = ("classifier" in n)
    args_head = make_training_args(
        output_dir=os.path.join(args.out_dir, "stage1_head"),
        learning_rate=2e-4,
        warmup_ratio=0.0,
        weight_decay=0.0,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size*2),
        num_train_epochs=args.epochs_head,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        save_total_limit=1,
        max_grad_norm=1.0,
    )
    trainer1 = BalancedTrainer(
        train_labels=train_labels, focal_alpha=alpha, focal_gamma=2.0,
        model=model, args=args_head,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        tokenizer=tokenizer, data_collator=collator,
        compute_metrics=compute_metrics
    )
    trainer1.train()
    print("✅ 阶段1完成")

    # ========= 阶段2：解冻全模型（标准采样 + Logit-Adjusted CE）=========
    for p_ in model.parameters():
        p_.requires_grad = True
    args_full = make_training_args(
        output_dir=os.path.join(args.out_dir, "stage2_full"),
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size*2),
        num_train_epochs=args.epochs_full,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        save_total_limit=1,
        max_grad_norm=0.5,
    )
    trainer2 = LogitAdjustedCETrainer(
        logit_bias=logit_bias,
        model=model, args=args_full,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        tokenizer=tokenizer, data_collator=collator,
        compute_metrics=make_compute_metrics_with_bias(logit_bias.astype(np.float32))
    )
    trainer2.train()
    metrics = trainer2.evaluate()
    print("✅ 最终评估:", metrics)

    # 保存与报告
    trainer2.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id":label2id,"id2label":id2label}, f, ensure_ascii=False, indent=2)
    print(f"✅ 保存到 {args.out_dir}")

    # 评估报告（同样加 bias）
    preds = trainer2.predict(ds["validation"])
    y_true = preds.label_ids
    logits = preds.predictions + logit_bias[None, :]
    y_pred = np.argmax(logits, axis=-1)
    print("\n[分类报告]\n", classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(num_labels)], digits=4))
    print("[混淆矩阵]\n", confusion_matrix(y_true, y_pred))

    # 自测（也做同样的后验修正）
    model.eval()
    device = next(model.parameters()).device
    for s in ["今天心情很好，效率也很高！", "真是气死我了，服务太差。", "一般般吧，没什么感觉。"]:
        enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=args.max_len)
        enc = {k:v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            out = model(**enc).logits + torch.tensor(logit_bias, device=device)
            pred = int(out.argmax(dim=-1).item())
        print(f"[自测] {s} -> {id2label[pred]}")

if __name__ == "__main__":
    main()
