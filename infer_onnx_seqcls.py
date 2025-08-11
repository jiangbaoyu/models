# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

LABEL2ID = {"negative":0, "neutral":1, "positive":2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def build_session(onnx_path: str, use_cuda: bool):
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("请先安装 onnxruntime 或 onnxruntime-gpu")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, providers=providers)

def predict(texts, onnx_path, tokenizer_path, max_len=128, use_cuda=False):
    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    sess = build_session(onnx_path, use_cuda)

    results = []
    for text in texts:
        enc = tok(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_len
        )
        if "token_type_ids" not in enc:
            enc["token_type_ids"] = np.zeros_like(enc["input_ids"])

        logits = sess.run(None, {
            "input_ids": enc["input_ids"].astype("int64"),
            "attention_mask": enc["attention_mask"].astype("int64"),
            "token_type_ids": enc["token_type_ids"].astype("int64")
        })[0]  # [1,3]

        probs = softmax(logits)
        pred_id = int(np.argmax(probs, axis=1)[0])
        results.append({
            "text": text,
            "pred": ID2LABEL[pred_id],
            "probs": {ID2LABEL[i]: float(probs[0, i]) for i in range(len(ID2LABEL))}
        })
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="导出的ONNX路径(带分类头)")
    ap.add_argument("--tokenizer", required=True, help="tokenizer目录（与你的微调权重一致）")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    if not Path(args.onnx).exists():
        raise SystemExit(f"未找到 ONNX 文件：{args.onnx}")
    if not Path(args.tokenizer).exists():
        raise SystemExit(f"未找到 tokenizer 目录：{args.tokenizer}")

    demo_texts = [
        "今天心情很好，效率也很高！",
        "真是气死我了，服务太差。",
        "有点恶心，不想再看第二眼。",
        "最近有点丧，提不起精神。"
    ]
    out = predict(
        texts=demo_texts,
        onnx_path=args.onnx,
        tokenizer_path=args.tokenizer,
        max_len=args.max_len,
        use_cuda=args.cuda
    )
    for r in out:
        print("文本:", r["text"])
        print("预测:", r["pred"])
        print("概率:", r["probs"])
        print("-"*60)

if __name__ == "__main__":
    main()