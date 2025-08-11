import torch
import torch.onnx
from transformers import BertTokenizer, BertForSequenceClassification
import os

def convert_finetuned_to_onnx():
    """将微调后的TinyBERT模型转换为ONNX格式"""
    
    # ===== 1. 检查微调后的模型是否存在 =====
    model_path = "./TinyBERT/tinybert_emotion_finetuned"
    if not os.path.exists(model_path):
        print("❌ 未找到微调后的模型")
        print("请先运行 finetune_tinybert.py 进行模型微调")
        return False
    
    # ===== 2. 加载微调后的模型和tokenizer =====
    print("📥 加载微调后的模型...")
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False
    
    # 设置模型为评估模式
    model.eval()
    
    # ===== 3. 准备虚拟输入 =====
    print("🔧 准备虚拟输入数据...")
    dummy_text = "这是一个测试文本"
    max_length = 128
    
    # 使用tokenizer处理虚拟输入
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    
    dummy_input_ids = inputs['input_ids']
    dummy_attention_mask = inputs['attention_mask']
    dummy_token_type_ids = inputs['token_type_ids']
    
    # ===== 4. 定义输入和输出名称 =====
    input_names = ['input_ids', 'attention_mask', 'token_type_ids']
    output_names = ['logits']
    
    # ===== 5. 定义动态轴（支持不同序列长度和批次大小） =====
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'token_type_ids': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size'}
    }
    
    # ===== 6. 导出ONNX模型 =====
    onnx_path = "tinybert_emotion_finetuned.onnx"
    print(f"🚀 开始导出ONNX模型到 {onnx_path}...")
    
    try:
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
            onnx_path,
            export_params=True,
            opset_version=14,  # 使用opset 14以支持更多操作
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        print("✅ ONNX模型导出成功！")
        
        # ===== 7. 验证ONNX模型 =====
        print("🔍 验证ONNX模型...")
        import onnxruntime as ort
        
        # 加载ONNX模型
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
        # 准备输入数据
        onnx_inputs = {
            'input_ids': dummy_input_ids.numpy(),
            'attention_mask': dummy_attention_mask.numpy(),
            'token_type_ids': dummy_token_type_ids.numpy()
        }
        
        # 运行推理
        onnx_outputs = session.run(None, onnx_inputs)
        
        # 与PyTorch模型输出对比
        with torch.no_grad():
            pytorch_outputs = model(dummy_input_ids, dummy_attention_mask, dummy_token_type_ids)
        
        # 检查输出是否一致
        pytorch_logits = pytorch_outputs.logits.numpy()
        onnx_logits = onnx_outputs[0]
        
        max_diff = abs(pytorch_logits - onnx_logits).max()
        print(f"📊 PyTorch与ONNX输出最大差异: {max_diff:.6f}")
        
        if max_diff < 1e-4:
            print("✅ ONNX模型验证通过！")
        else:
            print("⚠️  ONNX模型输出与PyTorch有较大差异")
        
        # ===== 8. 输出模型信息 =====
        print("\n📋 ONNX模型信息:")
        print(f"  - 文件路径: {onnx_path}")
        print(f"  - 文件大小: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")
        print(f"  - 输入数量: {len(session.get_inputs())}")
        print(f"  - 输出数量: {len(session.get_outputs())}")
        
        for i, input_info in enumerate(session.get_inputs()):
            print(f"  - 输入{i+1}: {input_info.name}, 形状: {input_info.shape}, 类型: {input_info.type}")
        
        for i, output_info in enumerate(session.get_outputs()):
            print(f"  - 输出{i+1}: {output_info.name}, 形状: {output_info.shape}, 类型: {output_info.type}")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX导出失败: {e}")
        return False

if __name__ == "__main__":
    print("🎯 TinyBERT情感分析模型 - 微调版本转ONNX")
    print("=" * 50)
    
    success = convert_finetuned_to_onnx()
    
    if success:
        print("\n🎉 转换完成！")
        print("现在您可以使用微调后的ONNX模型进行高精度情感分析了！")
    else:
        print("\n💡 提示:")
        print("1. 请先运行 'python finetune_tinybert.py' 进行模型微调")
        print("2. 确保已安装所需依赖: pip install torch transformers onnxruntime")