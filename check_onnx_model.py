# -*- coding: utf-8 -*-
import onnxruntime as ort
from pathlib import Path

def check_onnx_model(onnx_path):
    """检查ONNX模型的输入输出信息"""
    if not Path(onnx_path).exists():
        print(f"❌ ONNX文件不存在: {onnx_path}")
        return
    
    try:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
        print(f"✅ 成功加载ONNX模型: {onnx_path}")
        print("\n📥 输入信息:")
        for i, input_info in enumerate(sess.get_inputs()):
            print(f"  {i+1}. 名称: {input_info.name}")
            print(f"     类型: {input_info.type}")
            print(f"     形状: {input_info.shape}")
        
        print("\n📤 输出信息:")
        for i, output_info in enumerate(sess.get_outputs()):
            print(f"  {i+1}. 名称: {output_info.name}")
            print(f"     类型: {output_info.type}")
            print(f"     形状: {output_info.shape}")
            
    except Exception as e:
        print(f"❌ 加载ONNX模型失败: {e}")

if __name__ == "__main__":
    # 检查导出的带分类头ONNX模型
    check_onnx_model("./TinyBERT/tinybert_emotion_cls_128.onnx")
    print("\n" + "="*60 + "\n")
    # 对比原来的ONNX模型
    check_onnx_model("./TinyBERT/tinybert_emotion_analysis_128.onnx")