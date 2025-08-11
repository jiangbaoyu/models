# -*- coding: utf-8 -*-
"""
make_new_train.py
构建 new_train.json：
- 读取 CSV/JSON（可多文件）
- 清洗、去重、长度裁剪
- 有/无标签两种模式：
  (a) 有标签：用现有模型做置信度评估，挖掘难样本（低 margin / 高 entropy）
  (b) 无标签：自动打标，并过滤低置信样本
- 类均衡：--target-per-class
- 小类轻量增强（可选）：--augment
- 导出：
    new_train.json（训练集）
    review_hard.json（建议人工复核）
    stats.json（数据与筛选统计）
"""

import os, json, argparse, re, glob, uuid
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =============== 文本清洗 ===============
_URL = re.compile(r'https?://\S+|www\.\S+')
_AT = re.compile(r'@\S+')
_TOPIC = re.compile(r'#([^#]+)#')
_SPACE = re.compile(r'\s+')

def clean_text(s: str) -> str:
    s = str(s)
    s = _URL.sub(' ', s)
    s = _AT.sub(' ', s)
    s = _topic_sub(s)
    s = s.replace('转发微博', ' ').replace('来自', ' ')
    s = _SPACE.sub(' ', s).strip()
    return s

def _topic_sub(s: str) -> str:
    return _TOPIC.sub(r'\1', s)


# =============== 读数据（支持 CSV/JSON，多文件） ===============
TYPICAL_LABELS = {
    "angry","anger","happy","joy","sad","depress","fear","neutral","surprise","disgust",
    "生气","愤怒","高兴","开心","快乐","伤心","难过","恐惧","中性","惊讶","厌恶"
}

def load_one(path: str) -> pd.DataFrame:
    if path.lower().endswith((".json", ".jsonl", ".txt")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 兼容 content / text
        rows = []
        for it in data:
            text = it.get("content", it.get("text", None))
            if text is None:
                continue
            lab = it.get("label", None)
            rows.append({"text": text, "label": lab})
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path, sep=None, engine="python", header=None, keep_default_na=False)
        # 猜测 label 列
        def is_label_col(col) -> bool:
            vals = df[col].astype(str).str.strip().tolist()
            uniq = set(vals)
            if len(uniq & TYPICAL_LABELS) >= 3:
                return True
            try:
                nums = pd.to_numeric(vals, errors="coerce")
                u = set(nums.dropna().astype(int).unique().tolist())
                if 2 <= len(u) <= 20:
                    return True
            except Exception:
                pass
            return False

        label_cols = [c for c in df.columns if is_label_col(c)]
        if label_cols:
            lc = label_cols[0]
            text_cols = [c for c in df.columns if c != lc]
            text = df[text_cols].astype(str).apply(
                lambda r: " ".join([x for x in r if x and x.lower()!="nan"]), axis=1
            )
            df = pd.DataFrame({"text": text, "label": df[lc].astype(str)})
        else:
            # 无法识别标签列，当作纯文本
            text = df.astype(str).apply(
                lambda r: " ".join([x for x in r if x and x.lower()!="nan"]), axis=1
            )
            df = pd.DataFrame({"text": text, "label": None})
    # 清洗
    df["text"] = df["text"].astype(str).apply(clean_text)
    # 去空与重复
    df = df[df["text"].str.len() >= 1].drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df

def load_many(paths: List[str]) -> pd.DataFrame:
    files = []
    for p in paths:
        files.extend(glob.glob(p))
    if not files:
        raise FileNotFoundError("找不到任何输入文件")
    parts = [load_one(p) for p in files]
    df = pd.concat(parts, axis=0).reset_index(drop=True)
    return df


# =============== 模型推理（置信度/打标/难样本挖掘） ===============
@torch.no_grad()
def model_scores(texts: List[str], tok, model, max_len: int, batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """返回 logits(np.array [N,C]) 与 prob(np.array [N,C])"""
    device = next(model.parameters()).device
    all_logits = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, truncation=True, max_length=max_len, return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        all_logits.append(logits.cpu())
    logits = torch.cat(all_logits, dim=0).numpy()
    prob = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    return logits, prob

def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1.0)
    return -(p * np.log(p)).sum(axis=-1)

def margin(p: np.ndarray) -> np.ndarray:
    part = -np.sort(-p, axis=-1)  # 降序
    return part[:,0] - part[:,1]


# =============== 轻量“安全”增强（对小类） ===============
ADV_WORDS = [
    ("很", ["很","特别","格外","相当"]),
    ("非常", ["非常","十分","极其"]),
    ("有点", ["有点","有些","略微"]),
]
TAIL_PUNC = ["！","!!","！！！","~","～～","~!","!~"]

def light_augment(text: str, max_len: int) -> str:
    s = text
    # 1) 程度副词替换（保持语义强度相近）
    for key, candidates in ADV_WORDS:
        if key in s and np.random.rand() < 0.5:
            s = s.replace(key, np.random.choice(candidates), 1)
    # 2) 句尾情感符号增强
    if len(s) <= max_len and np.random.rand() < 0.4 and not s.endswith(("。","？","！","!","~")):
        s = s + np.random.choice(TAIL_PUNC)
    # 3) 轻微拉长音
    if np.random.rand() < 0.3:
        s = re.sub(r"(哈|啊|哦|呀|呜|啦){1}", lambda m: m.group(1) * np.random.randint(2,4), s, count=1)
    return s[:max_len]


# =============== 主流程 ===============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="输入文件（CSV/JSON；可多文件/通配）")
    ap.add_argument("--model-dir", type=str, required=True, help="已训练模型目录（含 tokenizer）")
    ap.add_argument("--out", type=str, default="new_train.json", help="输出训练集 JSON")
    ap.add_argument("--review-out", type=str, default="review_hard.json", help="输出建议复核的难样本")
    ap.add_argument("--stats-out", type=str, default="stats.json", help="输出统计信息")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--target-per-class", type=int, default=4000, help="每类目标样本上限（均衡）")
    ap.add_argument("--batch-size", type=int, default=64)

    # 模式选择
    ap.add_argument("--auto-label", action="store_true", help="无标签输入：用模型自动打标")
    ap.add_argument("--conf-thres", type=float, default=0.5, help="自动打标模式下的最小置信度阈值")
    ap.add_argument("--mine-hard", action="store_true", help="有标签输入：挖掘难样本以供人工复核")
    ap.add_argument("--hard-topk", type=int, default=2000, help="难样本导出数量上限")
    ap.add_argument("--hard-margin", type=float, default=0.2, help="margin < 阈值视为不确定")
    ap.add_argument("--augment", action="store_true", help="对小类做轻量增强")
    ap.add_argument("--augment-max-per-class", type=int, default=1000, help="每类最多增强多少条")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    np.random.seed(args.seed)

    # 1) 读取
    df = load_many(args.inputs)
    df = df[df["text"].str.len() >= args.min_len].copy().reset_index(drop=True)

    # 2) 模型与标签空间
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    id2label = model.config.id2label
    # 某些版本 id2label 的键是字符串下标
    id2label = {int(k): v for k, v in id2label.items()} if isinstance(list(model.config.id2label.keys())[0], str) else id2label
    label2id = {v: k for k, v in id2label.items()}
    known_labels = set(label2id.keys())

    # 3) 有/无标签处理
    has_label = df["label"].notna().sum() > 0
    auto_label_mode = args.auto_label or not has_label

    stats = {"total_in": len(df), "auto_label": bool(auto_label_mode)}

    if auto_label_mode:
        # 自动打标
        logits, prob = model_scores(df["text"].tolist(), tokenizer, model, args.max_len, args.batch_size)
        pred_id = prob.argmax(axis=-1)
        conf = prob.max(axis=-1)
        df["label"] = [id2label[int(i)] for i in pred_id]
        df["conf"] = conf
        # 过滤低置信
        keep = df["conf"] >= float(args.conf_thres)
        df_keep = df[keep].copy()
        stats.update({
            "auto_conf_thres": float(args.conf_thres),
            "auto_labeled": int(len(df_keep)),
            "auto_dropped_low_conf": int((~keep).sum())
        })
    else:
        # 有标签：只做模型评估，挖掘难样本
        # 将非受支持标签映射/剔除
        mask_known = df["label"].astype(str).isin(known_labels)
        dropped_unknown = int((~mask_known).sum())
        df = df[mask_known].copy().reset_index(drop=True)
        stats["dropped_unknown_label"] = dropped_unknown

        logits, prob = model_scores(df["text"].tolist(), tokenizer, model, args.max_len, args.batch_size)
        pred_id = prob.argmax(axis=-1)
        pred_label = [id2label[int(i)] for i in pred_id]
        conf = prob.max(axis=-1)
        mg = margin(prob)
        ent = entropy(prob)
        df["pred_label"] = pred_label
        df["conf"] = conf
        df["margin"] = mg
        df["entropy"] = ent

        # 难样本：1) 预测错的；2) margin < 阈值；按熵/低margin排序
        hard = df[(df["pred_label"] != df["label"]) | (df["margin"] < float(args.hard_margin))].copy()
        hard["score"] = (1 - hard["margin"]) + hard["entropy"]
        hard = hard.sort_values(by="score", ascending=False).head(int(args.hard_topk))
        # 导出复核集
        review = hard[["text","label","pred_label","conf","margin","entropy"]].to_dict("records")
        with open(args.review_out, "w", encoding="utf-8") as f:
            json.dump(review, f, ensure_ascii=False, indent=2)
        print(f"✅ 已导出建议复核的难样本：{args.review_out}（{len(review)} 条）")

        df_keep = df[["text","label","conf","margin","entropy"]].copy()

    # 4) 均衡 & 可选增强
    #   先裁剪每类到 target-per-class 上限；不足的类可做增强或过采样
    target = int(args.target_per_class)
    grouped = defaultdict(list)
    for _, row in df_keep.iterrows():
        grouped[str(row["label"])].append(row)

    # 顺序打乱，保证随机性
    for k in grouped:
        np.random.shuffle(grouped[k])

    final_rows = []
    augment_count = defaultdict(int)

    for lab, rows in grouped.items():
        rows = list(rows)
        # 裁剪过多
        if len(rows) > target:
            rows = rows[:target]
        # 不足：尝试增强或过采样
        if len(rows) < target:
            need = target - len(rows)
            base = rows.copy()
            if args.augment and len(base) > 0:
                aug_max = min(int(args.augment_max_per_class), need)
                # 轻量增强
                for i in range(aug_max):
                    src = base[i % len(base)]
                    new_text = light_augment(str(src["text"]), args.max_len)
                    rows.append({"text": new_text, "label": lab})
                augment_count[lab] += aug_max
                need -= aug_max
            # 若仍不足，简单过采样补齐
            while need > 0 and len(base) > 0:
                src = base[need % len(base)]
                rows.append({"text": src["text"], "label": lab})
                need -= 1

        # 收集
        for r in rows:
            final_rows.append({"id": str(uuid.uuid4()), "content": str(r["text"])[:args.max_len], "label": lab})

    # 5) 去重（再次）
    df_out = pd.DataFrame(final_rows).drop_duplicates(subset=["content","label"]).reset_index(drop=True)

    # 6) 导出
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(df_out.to_dict("records"), f, ensure_ascii=False, indent=2)

    # 7) 统计
    dist = Counter(df_out["label"].tolist())
    stats.update({
        "after_total": int(len(df_out)),
        "class_dist": dict(dist),
        "target_per_class": target,
        "augment_used": bool(args.augment),
        "augment_count": {k:int(v) for k,v in augment_count.items()}
    })
    with open(args.stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"✅ new_train 导出：{args.out}（{len(df_out)} 条）")
    print(f"📊 统计：{args.stats_out}")
    print("📌 类分布：", dict(dist))


if __name__ == "__main__":
    main()
