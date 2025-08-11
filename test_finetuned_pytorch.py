# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

ckpt = r"./TinyBERT/tinybert_emotion_finetuned"  # 你的微调目录
tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
m = AutoModelForSequenceClassification.from_pretrained(ckpt).eval()

def run(text):
    enc = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    if "token_type_ids" not in enc: 
        enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])
    with torch.no_grad():
        logits = m(**enc).logits
        probs = F.softmax(logits, dim=-1)[0].tolist()
    print(text, "->", probs)

for s in ["今天心情很好，效率也很高！","真是气死我了，服务太差。","有点恶心，不想再看第二眼。","最近有点丧，提不起精神。"]:
    run(s)