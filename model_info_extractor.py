# -*- coding: utf-8 -*-
"""
分析 TinyBERT 模型文件信息
由于无法安装 torch 库，此脚本提供模型信息分析和转换指导
"""
from __future__ import print_function
import os
import json
import struct


def analyze_pytorch_model(model_path):
    """分析 PyTorch 模型文件"""
    print("分析 PyTorch 模型文件: {}".format(model_path))
    
    if not os.path.exists(model_path):
        print("错误：模型文件不存在")
        return
    
    file_size = os.path.getsize(model_path)
    print("文件大小: {:.2f} MB".format(file_size / (1024 * 1024)))
    
    # 尝试读取文件头部信息
    try:
        with open(model_path, 'rb') as f:
            # 读取前 100 字节查看文件格式
            header = f.read(100)
            print("文件头部信息 (前20字节): {}".format(header[:20].hex()))
            
            # 检查是否为 pickle 格式
            f.seek(0)
            first_bytes = f.read(10)
            if first_bytes.startswith(b'\x80\x02'):
                print("检测到 Python pickle 格式")
            elif first_bytes.startswith(b'PK'):
                print("检测到 ZIP 格式 (PyTorch 1.6+)")
            else:
                print("未知格式")
                
    except Exception as e:
        print("读取文件时出错: {}".format(e))


def load_config(config_path):
    """加载并显示配置信息"""
    print("\n加载配置文件: {}".format(config_path))
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print("模型配置信息:")
        for key, value in config.items():
            print("  {}: {}".format(key, value))
        
        return config
    except Exception as e:
        print("读取配置文件时出错: {}".format(e))
        return None


def analyze_vocab(vocab_path):
    """分析词汇表文件"""
    print("\n分析词汇表文件: {}".format(vocab_path))
    
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        vocab_size = len(lines)
        print("词汇表大小: {} 个词汇".format(vocab_size))
        
        # 显示前10个词汇
        print("前10个词汇:")
        for i, line in enumerate(lines[:10]):
            print("  {}: {}".format(i, line.strip()))
            
        # 显示特殊词汇
        special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        print("\n特殊词汇位置:")
        for token in special_tokens:
            if token + '\n' in lines:
                idx = lines.index(token + '\n')
                print("  {}: 位置 {}".format(token, idx))
                
    except Exception as e:
        print("读取词汇表文件时出错: {}".format(e))


def print_conversion_guide():
    """打印转换指导"""
    print("\n" + "="*60)
    print("PyTorch 模型转 ONNX 指导")
    print("="*60)
    
    print("\n1. 安装必要的库:")
    print("   pip install torch transformers onnx")
    
    print("\n2. 使用 transformers 库转换 (推荐):")
    print("   from transformers import AutoModel, AutoTokenizer")
    print("   import torch")
    print("   ")
    print("   # 加载模型")
    print("   model = AutoModel.from_pretrained('./TinyBERT')")
    print("   tokenizer = AutoTokenizer.from_pretrained('./TinyBERT')")
    print("   ")
    print("   # 准备输入")
    print("   inputs = tokenizer('测试文本', return_tensors='pt')")
    print("   ")
    print("   # 导出 ONNX")
    print("   torch.onnx.export(model, (inputs['input_ids'],), 'model.onnx')")
    
    print("\n3. 在线转换工具:")
    print("   - Hugging Face Optimum: https://huggingface.co/docs/optimum")
    print("   - ONNX Model Zoo: https://github.com/onnx/models")
    
    print("\n4. 如果遇到网络问题:")
    print("   - 使用国内镜像: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple")
    print("   - 或下载离线安装包")
    
    print("\n5. 验证 ONNX 模型:")
    print("   import onnx")
    print("   model = onnx.load('model.onnx')")
    print("   onnx.checker.check_model(model)")


def main():
    model_dir = r"e:\\models\\TinyBERT"
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "pytorch_model.bin")
    vocab_path = os.path.join(model_dir, "vocab.txt")
    
    print("TinyBERT 模型分析工具")
    print("模型目录: {}".format(model_dir))
    
    # 分析配置文件
    config = load_config(config_path)
    
    # 分析模型权重文件
    analyze_pytorch_model(weights_path)
    
    # 分析词汇表
    if os.path.exists(vocab_path):
        analyze_vocab(vocab_path)
    
    # 打印转换指导
    print_conversion_guide()
    
    print("\n分析完成！")
    print("注意：要实际转换模型，需要安装 torch 和 transformers 库。")


if __name__ == "__main__":
    main()