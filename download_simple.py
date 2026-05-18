"""
简单的 Huatuo 数据集下载脚本
请在命令行中运行此脚本
"""
import os
import sys

# 必须在导入前设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'

from datasets import load_dataset

def main():
    print("=" * 60)
    print("Huatuo 医学数据集下载工具")
    print("=" * 60)
    print(f"使用镜像源: {os.environ['HF_ENDPOINT']}")
    print()
    
    # 数据集列表
    datasets_list = [
        'FreedomIntelligence/huatuo_knowledge_graph_qa',
        'FreedomIntelligence/huatuo_encyclopedia_qa',
        'FreedomIntelligence/huatuo_consultation_qa',
        'FreedomIntelligence/huatuo26M-testdatasets',
    ]
    
    print("可用数据集:")
    for i, name in enumerate(datasets_list, 1):
        print(f"  {i}. {name}")
    print()
    
    # 创建保存目录
    save_dir = "./huatuo_data"
    os.makedirs(save_dir, exist_ok=True)
    
    # 逐个下载
    for dataset_name in datasets_list:
        print(f"\n正在下载: {dataset_name}")
        print("-" * 60)
        try:
            ds = load_dataset(dataset_name, trust_remote_code=True)
            
            # 保存
            save_name = dataset_name.replace('/', '_')
            ds.save_to_disk(os.path.join(save_dir, save_name))
            
            # 也保存为JSON
            for split in ds:
                json_file = os.path.join(save_dir, f"{save_name}_{split}.json")
                ds[split].to_json(json_file)
            
            print(f"✓ 成功! 保存到: {save_dir}/{save_name}")
            for split in ds:
                print(f"  - {split}: {len(ds[split])} 条")
                
        except Exception as e:
            print(f"✗ 失败: {e}")
            print(f"  请手动访问: https://hf-mirror.com/datasets/{dataset_name}")
    
    print("\n" + "=" * 60)
    print("下载完成!")
    print("=" * 60)
    print(f"\n数据保存在: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    main()
