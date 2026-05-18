"""
检查微调前的准备工作是否到位
"""
import os

def check_model_files(model_path):
    """检查模型文件是否齐全"""
    required_files = [
        'config.json',
        'tokenizer.json', 
        'tokenizer_config.json',
        'generation_config.json',
        'model.safetensors.index.json'
    ]
    
    print(f"\n[检查模型文件]: {model_path}")
    all_exist = True
    
    for file in required_files:
        filepath = os.path.join(model_path, file)
        if os.path.exists(filepath):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - 缺失")
            all_exist = False
    
    # 检查权重文件
    safetensors = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
    if safetensors:
        print(f"  ✓ 权重文件: {len(safetensors)} 个")
        for sf in safetensors:
            size = os.path.getsize(os.path.join(model_path, sf)) / (1024 * 1024 * 1024)
            print(f"    - {sf} ({size:.2f} GB)")
    else:
        print("  ✗ 权重文件缺失")
        all_exist = False
    
    return all_exist

def check_dataset():
    """检查数据集是否齐全"""
    datasets = [
        ('huatuo_data/FreedomIntelligence_huatuo_knowledge_graph_qa/', '知识图谱问答'),
        ('huatuo_data/FreedomIntelligence_huatuo_encyclopedia_qa/', '百科问答'),
        ('huatuo_data/FreedomIntelligence_huatuo_consultation_qa/', '问诊问答'),
        ('data/medical_finetune.json', '医疗微调数据')
    ]
    
    print("\n[检查数据集]:")
    all_exist = True
    
    for path, name in datasets:
        full_path = os.path.join(os.getcwd(), path)
        if os.path.exists(full_path):
            if os.path.isdir(full_path):
                files = len([f for f in os.listdir(full_path) if f.endswith('.arrow')])
                print(f"  ✓ {name}: {files} 个文件")
            else:
                size = os.path.getsize(full_path) / (1024 * 1024)
                print(f"  ✓ {name}: {size:.2f} MB")
        else:
            print(f"  ✗ {name}: 缺失")
            all_exist = False
    
    return all_exist

def check_script_config():
    """检查训练脚本配置"""
    print("\n[检查训练脚本]:")
    
    script_path = 'train_lora_medical_terms.py'
    if os.path.exists(script_path):
        print(f"  ✓ {script_path} 存在")
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if 'model_name' in content:
            # 提取模型路径
            import re
            match = re.search(r'model_name = ["\'](.*?)["\']', content)
            if match:
                model_path = match.group(1)
                print(f"  ✓ 模型路径: {model_path}")
                
                # 检查路径是否存在
                if os.path.exists(model_path):
                    print(f"  ✓ 模型路径有效")
                else:
                    print(f"  ✗ 模型路径不存在")
                    return False
        else:
            print("  ✗ 未找到 model_name 配置")
            return False
        
        # 检查关键参数
        params = ['batch_size', 'epochs', 'learning_rate', 'lora_rank']
        for param in params:
            if param in content:
                print(f"  ✓ {param} 已配置")
            else:
                print(f"  ✗ {param} 缺失")
                return False
                
        return True
    else:
        print(f"  ✗ {script_path} 不存在")
        return False

def check_requirements():
    """检查依赖是否已安装"""
    print("\n[检查依赖环境]:")
    required_packages = [
        'torch',
        'transformers', 
        'peft',
        'datasets',
        'accelerate',
        'bitsandbytes'
    ]
    
    all_installed = True
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} - 未安装")
            all_installed = False
    
    # 检查 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠️ CUDA 不可用，将使用 CPU")
    except:
        print("  ✗ 无法检查 CUDA")
        
    return all_installed

def main():
    print("=" * 60)
    print("微调准备工作检查")
    print("=" * 60)
    
    # 检查模型
    model_path = './models/Qwen_Qwen2.5-3B-Instruct/Qwen/Qwen2___5-3B-Instruct'
    model_ok = check_model_files(model_path)
    
    # 检查数据集
    data_ok = check_dataset()
    
    # 检查脚本配置
    script_ok = check_script_config()
    
    # 检查依赖
    req_ok = check_requirements()
    
    # 汇总
    print("\n" + "=" * 60)
    print("检查结果汇总:")
    print("=" * 60)
    
    checks = [
        ("模型文件", model_ok),
        ("数据集", data_ok),
        ("脚本配置", script_ok),
        ("依赖环境", req_ok)
    ]
    
    all_ready = True
    for name, ok in checks:
        status = "✅ 就绪" if ok else "❌ 未就绪"
        print(f"  {name}: {status}")
        if not ok:
            all_ready = False
    
    if all_ready:
        print("\n🎉 所有准备工作已完成！可以开始微调训练了！")
        print("\n运行命令:")
        print("  python train_lora_medical_terms.py")
    else:
        print("\n⚠️  部分准备工作未完成，请检查上述项目")

if __name__ == "__main__":
    main()
