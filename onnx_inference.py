# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

# 根据实际模型的标签映射
LABEL2ID = {"negative":0, "neutral":1, "positive":2}
ID2LABEL = {v:k for k,v in LABEL2ID.items()}

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

def tokenize_batch(tokenizer, texts, max_len=64):
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="np"  # 直接得到 numpy，方便 ORT
    )
    # 有些 tokenizer 没有 token_type_ids，给一个全 0
    if "token_type_ids" not in enc:
        enc["token_type_ids"] = np.zeros_like(enc["input_ids"])
    return enc

def predict(texts, onnx_path="TinyBERT_emotion_quick.onnx", tokenizer_path="ckpt_quick", max_len=128, use_cuda=False, batch_size=1):
    # 1) 加载 tokenizer & ORT session
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    sess = build_session(onnx_path, use_cuda)
    
    # 模拟分类权重（实际应使用训练好的分类头）
    W = np.random.randn(312, 3)
    b = np.random.randn(3)

    results = []
    # 2) 逐个推理
    for text in texts:
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_len
        )
        
        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype("int64"),
            "attention_mask": inputs["attention_mask"].astype("int64"),
            "token_type_ids": inputs["token_type_ids"].astype("int64")
        }
        
        outputs = sess.run(None, onnx_inputs)
        cls_vector = outputs[1]  # CLS向量 [1, 312]
        
        # 计算logits
        logits = np.dot(cls_vector, W) + b
        probs = softmax(logits)
        pred_id = int(np.argmax(logits, axis=1)[0])
        
        results.append({
            "text": text,
            "pred": ID2LABEL[pred_id],
            "probs": {ID2LABEL[i]: float(probs[0][i]) for i in range(len(ID2LABEL))}
        })
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="TinyBERT_emotion_quick.onnx")
    ap.add_argument("--tokenizer", default="ckpt_quick")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--max_len", type=int, default=64)
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