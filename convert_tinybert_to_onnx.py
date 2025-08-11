# -*- coding: utf-8 -*-
"""
将本地 TinyBERT 的 PyTorch 权重 (pytorch_model.bin) 导出为 ONNX 模型。
模型目录: e:\\models\\TinyBERT
导出文件: e:\\models\\TinyBERT\\tinybert.onnx

注意：此脚本需要安装 torch、transformers 和 onnx 库
安装命令：pip install torch transformers onnx
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import onnx
import os





def main():
    # 加载模型和分词器
    model_dir = r"D:/models/emotion/TinyBERT"  # 本地 TinyBERT 模型路径
    
    print("加载本地 TinyBERT 模型和分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        print("模型加载成功")
    except Exception as e:
        print("分类模型加载失败，尝试加载基础模型: {}".format(e))
        try:
            from transformers import AutoModel
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModel.from_pretrained(model_dir)
            print("基础模型加载成功")
        except Exception as e2:
            print("模型加载失败: {}".format(e2))
            return
    
    model.eval()
    
    # 准备输入数据
    inputs = tokenizer("今天心情非常好！", return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # 导出模型为 ONNX 格式
    onnx_path = os.path.join(model_dir, "tinybert_emotion_analysis.onnx")
    
    print("开始导出 ONNX 模型...")
    try:
        torch.onnx.export(
             model, 
             (inputs['input_ids'],), 
             onnx_path,
             input_names=['input_ids'], 
             output_names=['output'], 
             opset_version=14
         )
        print("PyTorch 模型已导出为 ONNX 格式，保存在 {}".format(onnx_path))
        
        # 验证 ONNX 模型
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX 模型验证通过")
        except Exception as ve:
            print("ONNX 模型验证失败: {}".format(ve))
            
    except Exception as e:
        print("ONNX 导出失败: {}".format(e))
        print("请确保已安装所需库: pip install torch transformers onnx")


if __name__ == "__main__":
    main()