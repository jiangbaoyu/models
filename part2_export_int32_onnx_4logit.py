# -*- coding: utf-8 -*-
# 导出 ONNX（输入为 int32），图内自动 Cast 为 int64，避免 ORT/MindSpore Lite 的 int64 兼容问题
import os
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
from datetime import datetime

class Int32InputWrapper(nn.Module):
    """
    ONNX 导出包装器：
    - 接收 int32 输入 (input_ids / attention_mask / token_type_ids)
    - 在图里立刻 Cast 为 int64 后再喂给原模型
    这样导出的 ONNX 的 input dtype = int32，但模型内部仍按 int64 计算。
    """
    def __init__(self, model: nn.Module, need_token_type_ids: bool):
        super().__init__()
        self.model = model
        self.need_token_type_ids = need_token_type_ids

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor=None):
        # 强制转为 int64，保证与原模型一致
        input_ids = input_ids.to(dtype=torch.long)
        attention_mask = attention_mask.to(dtype=torch.long)
        if self.need_token_type_ids and token_type_ids is not None:
            token_type_ids = token_type_ids.to(dtype=torch.long)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits  # [B, num_labels]

def build_dummy_and_inputs(tokenizer, max_len: int) -> Tuple[Dict[str, torch.Tensor], List[str], bool]:
    enc = tokenizer(
        "用于ONNX导出的样例输入",
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_len
    )
    # 判断是否有 token_type_ids（如 BERT 有，DistilBERT 没有）
    need_tti = "token_type_ids" in enc

    # 用 int32 的 dummy，确保导出后的 ONNX 输入就是 int32
    inputs: Dict[str, torch.Tensor] = {
        "input_ids": enc["input_ids"].to(dtype=torch.int32),
        "attention_mask": enc["attention_mask"].to(dtype=torch.int32),
    }
    input_order = ["input_ids", "attention_mask"]
    if need_tti:
        inputs["token_type_ids"] = enc["token_type_ids"].to(dtype=torch.int32)
        input_order.append("token_type_ids")
    return inputs, input_order, need_tti

@torch.no_grad()
def torch_logits(model, tokenizer, texts, max_len: int, device: torch.device):
    model.eval().to(device)
    enc = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    # 原生模型用 int64
    feed = {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device)
    }
    if "token_type_ids" in enc:
        feed["token_type_ids"] = enc["token_type_ids"].to(device)
    return model(**feed).logits.cpu().numpy()

def ort_logits(onnx_path: str, tokenizer, texts, max_len: int, input_order: List[str]):
    # ORT 会按 ONNX 图定义的 dtype 要求输入；这里固定喂 int32，匹配我们导出的图
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out_name = sess.get_outputs()[0].name

    enc = tokenizer(texts, return_tensors="np", truncation=True, padding=True, max_length=max_len)
    feed = {}
    for k in input_order:
        if k in enc:
            arr = enc[k]
            if arr.dtype != np.int32:
                arr = arr.astype(np.int32, copy=False)
            feed[k] = arr
        else:
            if k == "token_type_ids":
                # 如果 tokenizer 没有返回，但导出的图需要，则补零（几乎不会发生，因为我们只在 need_tti=True 时导出）
                feed[k] = np.zeros_like(enc["input_ids"], dtype=np.int32)
            else:
                raise ValueError(f"缺少必要输入: {k}")

    return sess.run([out_name], feed)[0]

def export_int32_onnx(model_dir_or_name: str, output_path: str, max_len: int, opset: int, device: torch.device, num_labels: int):
    print(f"[Load] {model_dir_or_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name, use_fast=True)

    # 关键：强制 num_labels=目标维度，并允许忽略分类头尺寸不匹配（HF 会重建分类头）
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_dir_or_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    # 双重保险：确保 config 也同步
    base_model.config.num_labels = num_labels

    # 准备 int32 dummy 和输入顺序
    dummy_inputs, input_order, need_tti = build_dummy_and_inputs(tokenizer, max_len)

    # 包装器：把 int32 输入 Cast→int64 后调用原模型
    wrapper = Int32InputWrapper(base_model, need_token_type_ids=need_tti).to(device).eval()

    dynamic_axes = {k: {0: "batch", 1: "sequence"} for k in input_order}
    dynamic_axes["logits"] = {0: "batch"}

    # 在输出路径中添加时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name, ext = os.path.splitext(output_path)
    output_path_with_time = f"{base_name}_{timestamp}{ext}"
    
    os.makedirs(os.path.dirname(output_path_with_time) or ".", exist_ok=True)
    print(f"[Export] 导出 ONNX（输入=INT32, 输出维度={num_labels}）到 {output_path_with_time}")
    torch.onnx.export(
        wrapper,
        tuple(dummy_inputs[k].to(device) for k in input_order),  # int32 dummy
        output_path_with_time,
        input_names=input_order,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=opset
    )
    onnx.checker.check_model(output_path_with_time)
    print("[Export] 导出完成并通过 onnx.checker 检查。")

    # 形状强校验：必须是 [*, num_labels]
    _sess = ort.InferenceSession(output_path_with_time, providers=["CPUExecutionProvider"])
    out0 = _sess.get_outputs()[0]
    print(f"[Check] ONNX 输出名={out0.name}，类型={out0.type}，形状={out0.shape}")
    if len(out0.shape) < 2 or (out0.shape[-1] != num_labels and out0.shape[-1] is not None):
        raise RuntimeError(f"导出的 ONNX 最后一维不是 {num_labels}，实际 {out0.shape}。请确认分类头是否被正确重建。")

    # 情绪标签映射（按你的 4 类顺序来；确保与训练一致）
    emotion_labels = ["愤怒", "厌恶", "恐惧", "快乐"]
    if len(emotion_labels) != num_labels:
        # 如需自定义，可以改成从文件读取
        emotion_labels = [f"label_{i}" for i in range(num_labels)]

    # 模拟测试1：积极情绪测试
    print("\n[模拟测试1] 积极情绪测试")
    positive_texts = [
        "今天心情特别好，工作效率很高！",
        "刚收到好消息，开心得不得了！",
        "和朋友聚餐很愉快，笑得肚子疼。"
    ]
    logits_pos = ort_logits(output_path_with_time, tokenizer, positive_texts, max_len, input_order)
    for i, text in enumerate(positive_texts):
        pred_id = int(np.argmax(logits_pos[i]))
        confidence = float(np.exp(logits_pos[i][pred_id]) / np.sum(np.exp(logits_pos[i])))
        print(f"  文本: {text}")
        print(f"  预测: {emotion_labels[pred_id]} (置信度: {confidence:.3f})\n")
    
    # 模拟测试2：消极情绪测试
    print("[模拟测试2] 消极情绪测试")
    negative_texts = [
        "真是气死我了，服务态度太差了！",
        "这个产品质量太糟糕，完全不值这个价格。",
        "排队等了两个小时，结果告诉我没货了。"
    ]
    logits_neg = ort_logits(output_path_with_time, tokenizer, negative_texts, max_len, input_order)
    for i, text in enumerate(negative_texts):
        pred_id = int(np.argmax(logits_neg[i]))
        confidence = float(np.exp(logits_neg[i][pred_id]) / np.sum(np.exp(logits_neg[i])))
        print(f"  文本: {text}")
        print(f"  预测: {emotion_labels[pred_id]} (置信度: {confidence:.3f})\n")
    
    # 模拟测试3：中性/复杂情绪测试
    print("[模拟测试3] 中性/复杂情绪测试")
    neutral_texts = [
        "今天天气还可以，不冷不热的。",
        "这部电影一般般，没什么特别的感觉。",
        "工作完成了，准备下班回家。"
    ]
    logits_neu = ort_logits(output_path_with_time, tokenizer, neutral_texts, max_len, input_order)
    for i, text in enumerate(neutral_texts):
        pred_id = int(np.argmax(logits_neu[i]))
        confidence = float(np.exp(logits_neu[i][pred_id]) / np.sum(np.exp(logits_neu[i])))
        print(f"  文本: {text}")
        print(f"  预测: {emotion_labels[pred_id]} (置信度: {confidence:.3f})\n")
    
    # 校验：PyTorch vs ONNX top-1
    test_texts = [
        "今天心情很好，效率也很高！",
        "真是气死我了，服务太差。",
        "一般般吧，没什么感觉。"
    ]
    print("[Validate] 进行 PyTorch 与 ONNXRuntime 对齐校验...")
    logits_torch = torch_logits(base_model, tokenizer, test_texts, max_len, device)
    logits_onnx = ort_logits(output_path_with_time, tokenizer, test_texts, max_len, input_order)
    top1_ok = np.all(np.argmax(logits_torch, axis=1) == np.argmax(logits_onnx, axis=1))
    max_abs = float(np.max(np.abs(logits_torch - logits_onnx)))
    print(f"[Validate] top1一致={top1_ok} | max|Δ|={max_abs:.6f}")
    if not top1_ok:
        print("  [Warn] top1 不一致，通常是数值误差/激活边界所致；如需严格一致可尝试提高 opset 或关闭常量折叠。")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF 模型名或本地目录，例如 hfl/chinese-macbert-base 或 ./runs/finetune_dir")
    ap.add_argument("--output", required=True, help="ONNX 输出路径，例如 ./export/emotion_int32.onnx")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--opset", type=int, default=14)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--num_labels", type=int, default=4, help="分类类别数（默认 4）")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    export_int32_onnx(args.model, args.output, args.max_len, args.opset, device, args.num_labels)

if __name__ == "__main__":
    main()
