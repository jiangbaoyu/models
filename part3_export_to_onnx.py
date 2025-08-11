# export_to_onnx.py
# 用途：把HF情感分类模型导出为ONNX（可选强制INT32输入），并做ORT验证与可选简化
# 依赖：pip install torch transformers onnx onnxruntime onnxsim -U

import os
import argparse
import json
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def export_onnx(model_dir: str,
                onnx_path: str,
                max_len: int = 256,
                opset: int = 13,
                force_int32_inputs: bool = False,
                simplify: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 加载模型与分词器
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    # 2) 构造dummy输入（中文样例）
    dummy = tokenizer("今天心情很好，效率也很高！", return_tensors="pt", truncation=True, max_length=max_len)
    # 动态轴需要至少一些padding维度，确保导出图可扩展
    if dummy["input_ids"].shape[1] < 8:
        dummy = tokenizer("今天心情很好，效率也很高！"*8, return_tensors="pt",
                          truncation=True, max_length=max_len)

    # 可选：把输入cast为int32（多数NLP模型默认是int64，MSLite常对int64不友好）
    if force_int32_inputs:
        for k in ["input_ids", "attention_mask", "token_type_ids"]:
            if k in dummy:
                dummy[k] = dummy[k].to(torch.int32)
        input_dtypes = {"input_ids": "INT32", "attention_mask": "INT32",
                        "token_type_ids": "INT32" if "token_type_ids" in dummy else None}
    else:
        input_dtypes = {"input_ids": "INT64", "attention_mask": "INT64",
                        "token_type_ids": "INT64" if "token_type_ids" in dummy else None}

    # 移动到设备
    dummy = {k: v.to(device) for k, v in dummy.items()}

    # 3) 准备导出
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    input_names = ["input_ids", "attention_mask"]
    inputs = [dummy["input_ids"], dummy["attention_mask"]]
    dynamic_axes = {"input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "logits": {0: "batch"}}

    # 对有token_type_ids的模型（部分中文BERT系）一并导出
    if "token_type_ids" in dummy:
        input_names.append("token_type_ids")
        inputs.append(dummy["token_type_ids"])
        dynamic_axes["token_type_ids"] = {0: "batch", 1: "seq"}

    # 4) torch.onnx.export
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(inputs),
            onnx_path,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True
        )

    print(f"✅ ONNX导出成功: {onnx_path}")

    # 5) 可选：onnx-simplifier 简化
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            onnx_model = onnx.load(onnx_path)
            model_simplified, check = onnx_simplify(onnx_model, dynamic_input_shape=True)
            assert check, "ONNX Simplifier 校验失败"
            onnx.save(model_simplified, onnx_path)
            print("✅ ONNX 模型简化完成")
        except Exception as e:
            print(f"⚠️ 简化失败（可忽略）：{e}")

    # 6) ORT验证（对比PyTorch输出）
    try:
        import onnxruntime as ort

        sess_opt = ort.SessionOptions()
        sess = ort.InferenceSession(onnx_path, sess_options=sess_opt, providers=["CPUExecutionProvider"])

        # 准备一组测试句
        test_texts = [
            "今天心情很好，效率也很高！",
            "真是气死我了，服务太差。",
            "一般般吧，没什么感觉。",
            "有点害怕，心里直打鼓。",
        ]
        enc = tokenizer(test_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
        enc_torch = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            pt_logits = model(**enc_torch).logits.detach().cpu().numpy()

        # ORT需要numpy输入；且需与导出时dtype一致
        ort_inputs = {}
        for name in input_names:
            arr = enc[name].cpu().numpy()
            if force_int32_inputs:
                arr = arr.astype(np.int32, copy=False)
            else:
                # 某些tokenizer可能给int64，本就满足
                arr = arr.astype(np.int64, copy=False)
            ort_inputs[name] = arr

        ort_logits = sess.run(["logits"], ort_inputs)[0]

        # 归一化对比（允许极小数值误差）
        pt_prob = (torch.softmax(torch.tensor(pt_logits), dim=-1)).numpy()
        ort_prob = (torch.softmax(torch.tensor(ort_logits), dim=-1)).numpy()
        max_diff = float(np.max(np.abs(pt_prob - ort_prob)))
        print(f"🔍 ORT对齐检验: max|softmax(pt)-softmax(ort)| = {max_diff:.6f}")
        assert max_diff < 1e-4, "ORT与PyTorch输出差异偏大，请检查导出/简化步骤"

        # 打印IO与dtype信息
        print("📎 ONNX Inputs:")
        for i, inp in enumerate(sess.get_inputs()):
            print(f"  - {i}: name={inp.name}, shape={inp.shape}, dtype={inp.type}")
        out = sess.get_outputs()[0]
        print(f"📎 ONNX Output: name={out.name}, shape={out.shape}, dtype={out.type}")

        # 展示一条预测
        id2label = model.config.id2label
        sample_pred = ort_logits.argmax(axis=-1)[0]
        print(f"🧪 示例预测: \"{test_texts[0]}\" -> {id2label[int(sample_pred)]}")

        # 保存元信息
        meta = {
            "model_dir": model_dir,
            "onnx_path": onnx_path,
            "max_len": max_len,
            "opset": opset,
            "inputs_dtype": input_dtypes,
            "num_labels": int(model.config.num_labels),
            "id2label": {int(k): v for k, v in model.config.id2label.items()} if isinstance(list(model.config.id2label.keys())[0], int)
                        else {int(k): v for k, v in model.config.id2label.items()}
        }
        with open(os.path.splitext(onnx_path)[0] + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"✅ 元信息已写入: {os.path.splitext(onnx_path)[0]}.meta.json")

    except Exception as e:
        print(f"⚠️ ORT验证阶段出现问题（可先忽略再排查）：{e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF模型目录（如 ./runs/distilbert_balanced20k_retrain）")
    ap.add_argument("--onnx", type=str, required=True, help="导出的ONNX路径（如 ./onnx/distilbert_emotion_256.onnx）")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--force-int32-inputs", action="store_true", help="将输入张量dtype强制为INT32（便于后续MSLite转换）")
    ap.add_argument("--simplify", action="store_true", help="导出后用 onnx-simplifier 简化模型")
    args = ap.parse_args()

    export_onnx(
        model_dir=args.model,
        onnx_path=args.onnx,
        max_len=args.max_len,
        opset=args.opset,
        force_int32_inputs=args.force_int32_inputs,
        simplify=args.simplify
    )


if __name__ == "__main__":
    main()
