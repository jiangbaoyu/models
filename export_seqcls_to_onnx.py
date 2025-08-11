# -*- coding: utf-8 -*-
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL2ID = {"negative":0, "neutral":1, "positive":2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return out.logits  # 只导出 logits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="微调后HF权重目录")
    ap.add_argument("--onnx", default="tinybert_emotion_cls_128.onnx")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"找不到权重目录: {ckpt}")

    tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    # 若权重里没写好映射，这里兜底同步一下
    model.config.label2id = LABEL2ID
    model.config.id2label = ID2LABEL
    model.eval()

    # 构造固定长度的哑输入
    dummy = tok(
        "test",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.max_len
    )
    # TinyBERT是BERT系，带token_type_ids；若某些模型无此键，补零即可
    if "token_type_ids" not in dummy:
        dummy["token_type_ids"] = torch.zeros_like(dummy["input_ids"])

    wrapper = ModelWrapper(model)

    # 导出，固定shape（便于端侧/推理部署）
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
            args.onnx,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            opset_version=args.opset,
            dynamic_axes=None  # 固定长度
        )

    print(f"✅ 已导出: {args.onnx}")
    print(f"   输入: input_ids/attention_mask/token_type_ids -> [1,{args.max_len}]")
    print("   输出: logits -> [1,3]")

if __name__ == "__main__":
    main()