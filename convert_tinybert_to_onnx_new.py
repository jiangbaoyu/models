import torch
from transformers import AutoModel, AutoTokenizer
import onnx
import os

def main():
    print("开始转换TinyBERT模型为ONNX格式...")
    
    # 模型配置
    model_name = "huawei-noah/TinyBERT_General_4L_312D"
    seq_len = 128  # 序列长度
    
    print(f"📥 加载模型: {model_name}")
    
    # 加载tokenizer和模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        print("✅ 模型和tokenizer加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 设置模型为评估模式
    model.eval()
    
    # 准备示例输入
    print(f"🔧 准备示例输入 (序列长度: {seq_len})...")
    dummy_inputs = tokenizer(
        "示例文本", 
        max_length=seq_len, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )
    
    print(f"  - input_ids shape: {dummy_inputs['input_ids'].shape}")
    print(f"  - attention_mask shape: {dummy_inputs['attention_mask'].shape}")
    print(f"  - token_type_ids shape: {dummy_inputs['token_type_ids'].shape}")
    
    # 输出路径
    output_path = "./TinyBERT/tinybert_emotion_analysis_128.onnx"
    
    print(f"🚀 开始导出ONNX模型到: {output_path}")
    
    try:
        torch.onnx.export(
            model,
            (dummy_inputs["input_ids"], dummy_inputs["attention_mask"], dummy_inputs["token_type_ids"]),
            output_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={  # 支持动态长度
                "input_ids": {1: "seq_len"},
                "attention_mask": {1: "seq_len"},
                "token_type_ids": {1: "seq_len"},
            },
            opset_version=14
        )
        print("✅ ONNX模型导出成功！")
        
    except Exception as e:
        print(f"❌ ONNX导出失败: {e}")
        return
    
    # 验证ONNX模型
    print("🔍 验证ONNX模型...")
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX模型验证通过")
        
        # 显示模型信息
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"\n📋 ONNX模型信息:")
        print(f"  - 文件路径: {output_path}")
        print(f"  - 文件大小: {file_size:.2f} MB")
        print(f"  - 输入数量: 3 (input_ids, attention_mask, token_type_ids)")
        print(f"  - 输出数量: 1 (last_hidden_state)")
        print(f"  - 支持动态序列长度: 是")
        print(f"  - 最大序列长度: {seq_len}")
        
    except Exception as e:
        print(f"❌ ONNX模型验证失败: {e}")
        return
    
    print("\n🎉 TinyBERT转ONNX完成！")
    print(f"现在您可以使用 {output_path} 进行推理了！")
    print("\n注意: 此模型输出的是last_hidden_state，如需情感分析，需要额外的分类层。")

if __name__ == "__main__":
    main()