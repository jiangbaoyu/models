# part2_retrain_weight.py
# 目的：在已有模型（如 ./runs/distilbert_balanced20k）基础上继续训练/再训练
# 特点：
# 1) 类权重 CrossEntropyLoss（自动按数据分布计算，支持温和平滑）
# 2) 可选加权采样 WeightedRandomSampler（二选一：采样 or 类权重；默认用类权重）
# 3) 早停 EarlyStoppingCallback（监控 macro-F1）
# 4) 兼容老/新版 transformers 的 TrainingArguments（make_training_args）
# 5) 自动对齐/继承 label2id 与 id2label，避免标签错位
# 6) 训练/评估完整中文报告 + 混淆矩阵 + 若干自测样例
# Python 3.9 兼容

import os, argparse, json, re, inspect
from collections import Counter
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer, set_seed
)
try:
    from transformers import EarlyStoppingCallback
    HAS_EARLY = True
except Exception:
    HAS_EARLY = False


# =========================
# 兼容老/新版 TrainingArguments
# =========================
def make_training_args(**kw):
    """
    - 新版(支持 evaluation_strategy/save_strategy)：按 epoch 评估&保存，支持 load_best_model_at_end
    - 老版：移除不支持参数，退回 evaluate_during_training + save_steps/eval_steps，
            并关闭 load_best_model_at_end，避免策略不匹配报错
    """
    sig = inspect.signature(TrainingArguments.__init__)
    params = set(sig.parameters.keys())

    desired_defaults = dict(
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )
    for k, v in desired_defaults.items():
        kw.setdefault(k, v)

    new_api = ("eval_strategy" in params) and ("save_strategy" in params)

    if not new_api:
        # 老API：删掉不存在的键
        kw.pop("eval_strategy", None)
        kw.pop("save_strategy", None)
        for k in ["report_to", "metric_for_best_model", "greater_is_better",
                  "warmup_ratio", "save_total_limit", "max_grad_norm"]:
            if k not in params:
                kw.pop(k, None)

        if "evaluate_during_training" in params:
            kw["evaluate_during_training"] = True
        if "save_steps" in params and "save_steps" not in kw:
            kw["save_steps"] = 500
        if "eval_steps" in params and "eval_steps" not in kw:
            kw["eval_steps"] = kw.get("save_steps", 500)
        if "load_best_model_at_end" in params:
            kw["load_best_model_at_end"] = False

    filtered = {k: v for k, v in kw.items() if k in params}
    return TrainingArguments(**filtered)


# =========================
# 数据加载与清洗（CSV / JSON）
# =========================
def _clean_text(s: str) -> str:
    _url = re.compile(r'https?://\S+|www\.\S+')
    _at = re.compile(r'@\S+')
    _topic = re.compile(r'#([^#]+)#')
    _space = re.compile(r'\s+')
    s = _url.sub(' ', str(s))
    s = _at.sub(' ', s)
    s = _topic.sub(r'\1', s)
    s = s.replace('转发微博', ' ').replace('来自', ' ')
    s = _space.sub(' ', s).strip()
    return s

def load_json_data(json_path: str) -> pd.DataFrame:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts, labels = [], []
    # 允许 label 为 str/int；字段名 content/text 二选一
    for it in data:
        txt = it.get("content", it.get("text", ""))
        lab = it.get("label")
        if txt is None or lab is None:
            continue
        texts.append(txt)
        labels.append(lab)
    df = pd.DataFrame({"text": texts, "label": labels})
    df["text"] = df["text"].astype(str).apply(_clean_text)
    df = df.dropna().reset_index(drop=True)
    return df

def load_csv_any(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=None, engine="python", header=None, keep_default_na=False)
    df = df.replace(r"^\s*$", np.nan, regex=True).dropna(how="all").reset_index(drop=True)

    # 自动识别 label 列（整数/字符串都可）
    # 若存在典型情感标签列，则优先该列
    def is_label_col(col):
        vals = df[col].astype(str).str.strip().tolist()
        uniq = set(vals)
        # 常见中文/英文情感关键词/或有限整数集合
        typical = {"anger","angry","happy","joy","sad","depress","fear","neutral","surprise","disgust",
                   "生气","愤怒","高兴","开心","快乐","伤心","难过","恐惧","中性","惊讶","厌恶"}
        if len(uniq & typical) >= 3:
            return True
        # 或者是小整数标签
        try:
            nums = pd.to_numeric(vals, errors="coerce")
            uv = set(nums.dropna().astype(int).unique().tolist())
            if 2 <= len(uv) <= 20:
                return True
        except Exception:
            pass
        return False

    label_candidates = [c for c in df.columns if is_label_col(c)]
    if not label_candidates:
        raise ValueError("CSV 未能识别标签列，请检查数据。")
    label_col = label_candidates[0]
    text_cols = [c for c in df.columns if c != label_col]
    if not text_cols:
        raise ValueError("CSV 未找到文本列。")

    text = (
        df[text_cols].astype(str)
        .apply(lambda r: " ".join([x for x in r if x and x.lower() != "nan"]), axis=1)
        .astype(str).apply(_clean_text)
    )
    out = pd.DataFrame({"label": df[label_col].astype(str), "text": text})
    out = out[out["text"].str.len().ge(2)].drop_duplicates(subset=["label","text"]).reset_index(drop=True)
    return out

def load_dataset_any(path: str, seed: int, test_size=0.2, limit_train=None, limit_val=None):
    if path and os.path.exists(path):
        if path.endswith(".json") or path.endswith(".txt"):
            df = load_json_data(path)
        else:
            df = load_csv_any(path)
    else:
        # fallback 小样本
        texts  = ["今天心情很好，效率也很高！","真是气死我了，服务太差。","一般般吧，没什么感觉。","这个功能做得不错，体验很棒！","糟透了，再也不会用了。","还行吧，中规中矩。"]
        labels = ["joy","anger","neutral","joy","anger","neutral"]
        df = pd.DataFrame({"text":texts, "label":labels})

    # 统一为字符串标签
    df["label"] = df["label"].astype(str)
    # stratified split
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    if limit_train and len(train_df) > limit_train:
        train_df, _ = train_test_split(train_df, train_size=limit_train, random_state=seed, stratify=train_df["label"])
    if limit_val and len(val_df) > limit_val:
        val_df, _ = train_test_split(val_df, train_size=limit_val, random_state=seed, stratify=val_df["label"])

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# =========================
# Trainer（类权重）
# =========================
class WeightedCETrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.cw = torch.tensor(class_weights, dtype=torch.float)
            self.criterion = nn.CrossEntropyLoss(weight=self.cw)
        else:
            self.cw = None
            self.criterion = nn.CrossEntropyLoss()

    # 兼容不同版本 Trainer 的签名
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        if self.cw is not None:
            # 确保类权重在正确的设备上
            device = outputs.logits.device
            if self.cw.device != device:
                self.cw = self.cw.to(device)
                self.criterion = nn.CrossEntropyLoss(weight=self.cw)
        loss = self.criterion(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# =========================
# 评估指标
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }


# =========================
# 主流程
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="./runs/distilbert_balanced20k",
                    help="已训练模型目录（将作为再训练的初始化权重）")
    ap.add_argument("--data", type=str, default="./data/train/usual_train.json",
                    help="新一轮训练用的数据文件（JSON/CSV）")
    ap.add_argument("--out_dir", type=str, default="./runs/distilbert_balanced20k_retrain")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=8e-6, help="再训练通常用更小LR")
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.02)
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_val", type=int, default=None)
    ap.add_argument("--use_sampler", action="store_true",
                    help="是否改用 WeightedRandomSampler（与类权重二选一）")
    ap.add_argument("--class_smooth", type=float, default=0.05,
                    help="计算类权重时的拉普拉斯平滑系数，0~0.2 合理")
    ap.add_argument("--patience", type=int, default=2, help="早停轮数")
    ap.add_argument("--resume_from", type=str, default=None,
                    help="从某个 checkpoint 继续，如: runs/.../checkpoint-XXXX")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # 载入 tokenizer & 旧模型（继承 label2id/id2label）
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    base_model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    num_labels = int(base_model.config.num_labels)
    id2label = getattr(base_model.config, "id2label", {i: str(i) for i in range(num_labels)})
    label2id = getattr(base_model.config, "label2id", {v: k for k, v in id2label.items()})

    # 加载新数据
    train_df, val_df = load_dataset_any(args.data, seed=args.seed, limit_train=args.limit_train, limit_val=args.limit_val)

    # 若新数据的标签非同一集合，则建立新的映射（并重建分类头）
    uniq_labels = sorted(train_df["label"].astype(str).unique().tolist())
    if set(uniq_labels) != set(label2id.keys()):
        print("⚠️ 新数据的标签集合与旧模型不完全一致，将以新数据为准重建映射与分类头。")
        label2id = {lab: i for i, lab in enumerate(uniq_labels)}
        id2label = {i: lab for lab, i in label2id.items()}
        num_labels = len(label2id)
        # 重建分类头，但保留 encoder 权重
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir, num_labels=num_labels, id2label=id2label, label2id=label2id
        )
    else:
        # 使用原头
        model = base_model

    # tokenize
    def preprocess_texts(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["label_id"] = train_df["label"].map(label2id).astype(int)
    val_df["label_id"]   = val_df["label"].map(label2id).astype(int)

    ds = DatasetDict({
        "train": Dataset.from_pandas(train_df[["text","label_id"]].rename(columns={"label_id":"label"})),
        "validation": Dataset.from_pandas(val_df[["text","label_id"]].rename(columns={"label_id":"label"})),
    })
    keep_cols = ["input_ids","attention_mask","label"]
    ds = ds.map(preprocess_texts, batched=True).remove_columns(
        [c for c in ds["train"].column_names if c not in keep_cols]
    )

    collator = DataCollatorWithPadding(tokenizer)

    # 计算类权重（或采样权重）
    y_train = list(train_df["label_id"])
    cnt = Counter(y_train)
    K = len(cnt); N = len(y_train)
    # 拉普拉斯平滑，避免极小类权重爆炸
    alpha = float(args.class_smooth)
    freq = np.array([ (cnt[i] + alpha) / (N + alpha*K) for i in range(K) ], dtype=np.float32)
    class_weights = (1.0 / freq)
    # 归一化到均值=1
    class_weights = class_weights / class_weights.mean()
    print("类频次：", dict(cnt))
    print("类权重：", np.round(class_weights, 3).tolist())

    # dataloader：如果使用加权采样
    def make_train_loader(dataset):
        if not args.use_sampler:
            return None  # 走 Trainer 默认 DataLoader
        w_per_class = {i: 1.0 / max(1, cnt[i]) for i in range(K)}
        sample_w = [w_per_class[int(y)] for y in y_train]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_w, dtype=torch.double),
            num_samples=len(sample_w), replacement=True
        )
        num_workers = 0
        return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                          collate_fn=collator, num_workers=num_workers)

    # 训练参数
    args_train = make_training_args(
        output_dir=args.out_dir,
        learning_rate=float(args.lr),
        num_train_epochs=float(args.epochs),
        warmup_ratio=float(args.warmup_ratio),
        weight_decay=float(args.weight_decay),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size*2),
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # 自定义 Trainer
    trainer = WeightedCETrainer(
        class_weights=None if args.use_sampler else class_weights,  # 二选一
        model=model,
        args=args_train,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.patience)
        ] if HAS_EARLY else None
    )

    # 如果使用自定义 sampler，需要覆盖 get_train_dataloader
    if args.use_sampler:
        def _get_train_dataloader():
            dl = make_train_loader(trainer.train_dataset)
            return dl
        trainer.get_train_dataloader = _get_train_dataloader

    # 训练 / 续训
    trainer.train(resume_from_checkpoint=args.resume_from)

    # 评估
    metrics = trainer.evaluate()
    print("✅ 最终评估:", metrics)

    # 保存
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
    print(f"✅ 已保存到 {args.out_dir}")

    # 详细报告
    preds = trainer.predict(ds["validation"])
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(-1)
    names = [id2label[i] for i in range(len(id2label))]
    print("\n[分类报告]\n", classification_report(y_true, y_pred, target_names=names, digits=4))
    print("[混淆矩阵]\n", confusion_matrix(y_true, y_pred))

    # 自测
    model.eval()
    device = next(model.parameters()).device
    samples = [
        "今天心情很好，效率也很高！",
        "真是气死我了，服务太差。",
        "一般般吧，没什么感觉。",
        "有点恶心，不想再看第二眼。",
        "有点害怕，心里直打鼓。"
    ]
    for s in samples:
        enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=args.max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc).logits
            pred = int(out.argmax(dim=-1).item())
        print(f"[自测] {s} -> {id2label[pred]}")


if __name__ == "__main__":
    main()
