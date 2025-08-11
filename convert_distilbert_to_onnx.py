import torch
import torch.onnx
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

def convert_distilbert_to_onnx():
    """将DistilBERT中文模型转换为ONNX格式用于情感分析"""
    
    print("🎯 DistilBERT中文模型转ONNX")
    print("=" * 50)
    
    # ===== 1. 检查模型文件是否存在 =====
    model_path = "./distilbert-base-zh-cased"
    if not os.path.exists(model_path):
        print("❌ 未找到DistilBERT模型文件")
        print(f"请确保模型文件存在于: {model_path}")
        return False
    
    # ===== 2. 加载tokenizer =====
    print("📥 加载tokenizer...")
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        print("✅ Tokenizer加载成功")
    except Exception as e:
        print(f"❌ Tokenizer加载失败: {e}")
        return False
    
    # ===== 3. 创建分类模型 =====
    print("🔧 创建情感分析模型...")
    try:
        # 创建一个3分类的情感分析模型
        model = DistilBertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=3,  # 负面、中性、正面
            id2label={0: "负面", 1: "中性", 2: "正面"},
            label2id={"负面": 0, "中性": 1, "正面": 2}
        )
        model.eval()
        print("✅ 模型创建成功")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False
    
    # ===== 4. 准备虚拟输入 =====
    print("🔧 准备虚拟输入数据...")
    dummy_text = "今天天气很好，心情不错"
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
    
    print(f"  - 输入文本: {dummy_text}")
    print(f"  - 序列长度: {max_length}")
    print(f"  - input_ids形状: {dummy_input_ids.shape}")
    print(f"  - attention_mask形状: {dummy_attention_mask.shape}")
    
    # ===== 5. 定义输入和输出名称 =====
    input_names = ['input_ids', 'attention_mask']
    output_names = ['logits']
    
    # ===== 6. 定义动态轴 =====
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size'}
    }
    
    # ===== 7. 导出ONNX模型 =====
    onnx_path = "./distilbert-base-zh-cased/distilbert_emotion_analysis.onnx"
    print(f"🚀 开始导出ONNX模型到 {onnx_path}...")
    
    try:
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        print("✅ ONNX模型导出成功！")
        
    except Exception as e:
        print(f"❌ ONNX导出失败: {e}")
        return False
    
    # ===== 8. 验证ONNX模型 =====
    print("🔍 验证ONNX模型...")
    try:
        import onnxruntime as ort
        
        # 加载ONNX模型
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
        # 准备输入数据
        onnx_inputs = {
            'input_ids': dummy_input_ids.numpy(),
            'attention_mask': dummy_attention_mask.numpy()
        }
        
        # 运行推理
        onnx_outputs = session.run(None, onnx_inputs)
        
        # 与PyTorch模型输出对比
        with torch.no_grad():
            pytorch_outputs = model(dummy_input_ids, dummy_attention_mask)
        
        # 检查输出是否一致
        pytorch_logits = pytorch_outputs.logits.numpy()
        onnx_logits = onnx_outputs[0]
        
        max_diff = abs(pytorch_logits - onnx_logits).max()
        print(f"📊 PyTorch与ONNX输出最大差异: {max_diff:.6f}")
        
        if max_diff < 1e-4:
            print("✅ ONNX模型验证通过！")
        else:
            print("⚠️  ONNX模型输出与PyTorch有较大差异")
        
        # ===== 9. 输出模型信息 =====
        print("\n📋 ONNX模型信息:")
        print(f"  - 文件路径: {onnx_path}")
        print(f"  - 文件大小: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")
        print(f"  - 输入数量: {len(session.get_inputs())}")
        print(f"  - 输出数量: {len(session.get_outputs())}")
        
        for i, input_info in enumerate(session.get_inputs()):
            print(f"  - 输入{i+1}: {input_info.name}, 形状: {input_info.shape}, 类型: {input_info.type}")
        
        for i, output_info in enumerate(session.get_outputs()):
            print(f"  - 输出{i+1}: {output_info.name}, 形状: {output_info.shape}, 类型: {output_info.type}")
        
        # ===== 10. 测试情感分析 =====
        print("\n🧪 测试情感分析功能:")
        test_texts = [
            "今天天气真好，心情特别棒！",
            "这部电影太糟糕了，完全浪费时间",
            "还可以吧，没什么特别的"
        ]
        
        id2label = {0: "负面", 1: "中性", 2: "正面"}
        
        for text in test_texts:
            # Tokenize
            test_inputs = tokenizer(
                text,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
            
            # ONNX推理
            test_onnx_inputs = {
                'input_ids': test_inputs['input_ids'],
                'attention_mask': test_inputs['attention_mask']
            }
            
            test_outputs = session.run(None, test_onnx_inputs)
            logits = test_outputs[0]
            
            # 获取预测结果
            import numpy as np
            
            # 手动实现softmax
            def softmax(x):
                exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
                return exp_x / np.sum(exp_x, axis=1, keepdims=True)
            
            predicted_class = int(np.argmax(logits, axis=1)[0])
            predicted_label = id2label[predicted_class]
            probabilities = softmax(logits)[0]
            confidence = float(probabilities[predicted_class])
            
            print(f"  📝 \"{text}\" → {predicted_label} (置信度: {confidence:.4f})")
        
        return True
        
    except ImportError:
        print("⚠️  未安装onnxruntime，跳过验证步骤")
        print("   可运行: pip install onnxruntime")
        return True
    except Exception as e:
        print(f"❌ ONNX验证失败: {e}")
        return False

if __name__ == "__main__":
    success = convert_distilbert_to_onnx()
    
    if success:
        print("\n🎉 DistilBERT转ONNX完成！")
        print("现在您可以使用 distilbert_emotion_analysis.onnx 进行高效推理了！")
    else:
        print("\n💡 提示:")
        print("1. 确保DistilBERT模型文件存在于 './distilbert-base-zh-cased' 目录")
        print("2. 确保已安装所需依赖: pip install torch transformers onnxruntime")