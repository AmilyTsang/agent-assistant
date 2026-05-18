"""
Huatuo 医学数据集下载脚本
支持多种下载方式和错误处理
"""
import os
import json
import sys

# ============================================
# 重要：必须在导入 datasets 之前设置环境变量
# ============================================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5分钟超时
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # 启用加速传输

# 现在导入 datasets
from datasets import load_dataset

def set_huggingface_mirror():
    """显示当前镜像配置"""
    print(f"已配置 Hugging Face 镜像源: {os.environ.get('HF_ENDPOINT')}")
    print(f"超时设置: {os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT')}秒")

def download_dataset_with_mirror(dataset_name, save_dir="./data"):
    """
    使用镜像源下载数据集
    
    Args:
        dataset_name: 数据集名称
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"正在下载数据集: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # 方式1: 直接从 Hugging Face Hub 加载（使用镜像）
        print("尝试从镜像源加载...")
        dataset = load_dataset(
            dataset_name,
            download_mode="reuse_dataset_if_exists",
            trust_remote_code=True
        )
        
        # 保存数据集
        save_path = os.path.join(save_dir, dataset_name.replace('/', '_'))
        dataset.save_to_disk(save_path)
        print(f"\n✓ 数据集已保存到: {save_path}")
        
        # 也保存为 JSON 格式（可选）
        for split in dataset:
            json_path = os.path.join(save_dir, f"{dataset_name.replace('/', '_')}_{split}.json")
            dataset[split].to_json(json_path)
            print(f"✓ JSON 格式已保存到: {json_path}")
        
        return dataset
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n建议:")
        print("1. 检查网络连接")
        print("2. 尝试手动从以下地址下载:")
        print(f"   https://hf-mirror.com/datasets/{dataset_name}")
        print("3. 或者使用代理")
        return None

def print_manual_download_guide():
    """打印手动下载指引"""
    print("\n" + "="*60)
    print("手动下载指引")
    print("="*60)
    print("\n如果自动下载失败，可以手动下载：")
    print("\n1. 访问 Hugging Face 镜像站:")
    print("   https://hf-mirror.com/FreedomIntelligence")
    print("\n2. 下载以下数据集:")
    datasets_list = [
        "huatuo_knowledge_graph_qa",
        "huatuo_encyclopedia_qa", 
        "huatuo_consultation_qa",
        "huatuo26M-testdatasets",
        "Huatuo26M-Lite"
    ]
    for ds in datasets_list:
        print(f"   - {ds}")
    print("\n3. 下载后放到 ./data/ 目录下")
    print("\n4. 使用以下代码加载本地数据集:")
    print("""
from datasets import load_from_disk
dataset = load_from_disk('./data/FreedomIntelligence_huatuo_knowledge_graph_qa')
    """)

def test_small_dataset():
    """测试下载一个小数据集"""
    print("\n" + "="*60)
    print("测试下载（小数据集）")
    print("="*60)
    
    try:
        # 先测试一个小的官方数据集
        print("测试下载 tiny_dataset...")
        test_ds = load_dataset("rotten_tomatoes", split="train[:10]")
        print(f"✓ 测试成功！下载了 {len(test_ds)} 条数据")
        print(f"  列名: {test_ds.column_names}")
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def main():
    # 在最开始就设置环境变量
    set_huggingface_mirror()
    
    # 先测试网络连接
    if not test_small_dataset():
        print("\n网络连接有问题，请检查后重试，或使用手动下载方式")
        print_manual_download_guide()
        sys.exit(1)
    
    # 要下载的数据集列表
    datasets_to_download = [
        'FreedomIntelligence/huatuo_knowledge_graph_qa',
        'FreedomIntelligence/huatuo_encyclopedia_qa', 
        'FreedomIntelligence/huatuo_consultation_qa',
        'FreedomIntelligence/huatuo26M-testdatasets',
        'FreedomIntelligence/Huatuo26M-Lite'
    ]
    
    # 逐个下载数据集
    downloaded_datasets = {}
    for dataset_name in datasets_to_download:
        dataset = download_dataset_with_mirror(dataset_name)
        if dataset is not None:
            downloaded_datasets[dataset_name] = dataset
    
    print("\n" + "="*60)
    print(f"下载完成！成功下载 {len(downloaded_datasets)}/{len(datasets_to_download)} 个数据集")
    print("="*60)
    
    # 显示数据集信息
    if downloaded_datasets:
        for name, ds in downloaded_datasets.items():
            print(f"\n✓ {name}")
            print(f"  分割: {list(ds.keys())}")
            for split in ds:
                print(f"    {split}: {len(ds[split])} 条")
                print(f"    列名: {ds[split].column_names}")
    
    # 如果有失败的，显示手动下载指引
    if len(downloaded_datasets) < len(datasets_to_download):
        print_manual_download_guide()

if __name__ == "__main__":
    main()