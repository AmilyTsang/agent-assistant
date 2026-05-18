"""
LoRA微调脚本：使用华图知识图谱数据集训练医疗术语解释模型
目标：构建能够解释医疗报告中术语的专业模型
"""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

class MedicalTermDataset(Dataset):
    """医疗术语解释数据集"""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建训练样本：术语解释格式
        instruction = item["instruction"]
        output_text = item["output"]
        
        # 构建提示模板
        prompt = f"解释医疗术语：{instruction}\n解释："
        full_text = prompt + output_text
        
        # 编码
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 计算prompt长度
        prompt_length = len(self.tokenizer.encode(prompt))
        
        # 标签：只对输出部分进行监督学习
        labels = encoding["input_ids"].clone()
        labels[0, :prompt_length] = -100  # 将prompt部分设为-100（忽略）
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": labels.flatten()
        }

def load_huatuo_knowledge_dataset():
    """加载华图知识图谱数据集"""
    print("正在加载华图医疗知识图谱数据集...")
    
    try:
        # 加载数据集
        dataset = load_dataset('FreedomIntelligence/huatuo_knowledge_graph_qa')
        
        # 提取训练数据并转换格式
        train_data = []
        for item in dataset['train']:
            # 构建术语解释样本
            train_data.append({
                "instruction": item.get('question', item.get('query', '')),
                "output": item.get('answer', item.get('response', ''))
            })
        
        print(f"成功加载 {len(train_data)} 条医疗术语解释样本")
        return train_data
    
    except Exception as e:
        print(f"加载数据集失败: {e}")
        # 如果加载失败，使用本地备用数据
        print("尝试使用本地数据集...")
        return load_local_medical_terms()

def load_local_medical_terms():
    """加载本地医疗术语数据集（备用）"""
    medical_terms = [
        {"instruction": "高血压", "output": "高血压是指动脉血压持续升高，收缩压≥140mmHg和/或舒张压≥90mmHg。长期高血压可导致心脏病、脑卒中等并发症。"},
        {"instruction": "糖尿病", "output": "糖尿病是一种代谢性疾病，特征是血糖水平持续升高。主要分为1型和2型，需要通过饮食控制、运动和药物治疗。"},
        {"instruction": "冠心病", "output": "冠心病是由于冠状动脉粥样硬化导致心肌缺血缺氧的疾病，常见症状为心绞痛、心肌梗死等。"},
        {"instruction": "脑卒中", "output": "脑卒中又称中风，是由于脑部血管阻塞或破裂导致脑组织损伤的疾病，可分为缺血性和出血性两种。"},
        {"instruction": "癌症", "output": "癌症是由细胞异常增生形成的恶性肿瘤，可发生在身体的任何部位，早期发现和治疗至关重要。"},
        {"instruction": "肺炎", "output": "肺炎是肺部的感染性疾病，通常由细菌、病毒或真菌引起，常见症状包括发热、咳嗽、呼吸困难等。"},
        {"instruction": "哮喘", "output": "哮喘是一种慢性气道炎症性疾病，特征是气道狭窄和呼吸困难，常由过敏原、感染或运动诱发。"},
        {"instruction": "关节炎", "output": "关节炎是关节的炎症性疾病，常见症状包括关节疼痛、肿胀和僵硬，严重时可导致关节畸形。"},
        {"instruction": "阿司匹林", "output": "阿司匹林是一种非甾体抗炎药，具有解热、镇痛、抗炎和抗血小板聚集作用。常用于缓解疼痛、降低体温、预防心脑血管疾病等。"},
        {"instruction": "布洛芬", "output": "布洛芬是一种非甾体抗炎药，具有解热、镇痛、抗炎作用。常用于缓解轻至中度疼痛，如头痛、关节痛、牙痛等。"},
        {"instruction": "CT扫描", "output": "CT扫描即计算机断层扫描，是一种利用X射线穿透人体并通过计算机处理形成断层图像的医学检查技术，可用于诊断多种疾病。"},
        {"instruction": "MRI", "output": "MRI即磁共振成像，是利用磁场和无线电波生成人体内部结构详细图像的检查方法，无辐射，对软组织成像效果好。"},
        {"instruction": "心电图", "output": "心电图是记录心脏电活动的检查方法，通过在体表放置电极来检测心脏的节律和心肌缺血情况，常用于诊断心律失常和心肌梗死。"},
        {"instruction": "血常规", "output": "血常规是通过检测血液中的红细胞、白细胞、血小板等指标来评估身体基本健康状况的检查，可发现感染、贫血等问题。"},
        {"instruction": "肝功能", "output": "肝功能检查通过检测血液中的转氨酶、胆红素、白蛋白等指标来评估肝脏的代谢和合成功能，用于诊断肝炎、肝硬化等疾病。"}
    ]
    return medical_terms

def train_lora_medical_terms():
    """执行医疗术语解释模型的LoRA微调"""
    
    # 参数配置
    model_name = "./models/Qwen_Qwen2.5-3B-Instruct/Qwen/Qwen2___5-3B-Instruct"
    output_dir = "./lora_medical_terms"
    lora_rank = 8
    batch_size = 2
    epochs = 3
    learning_rate = 3e-4
    max_length = 512
    
    print("=" * 60)
    print("医疗术语解释模型 LoRA 微调训练")
    print("=" * 60)
    
    # 加载数据集
    train_data = load_huatuo_knowledge_dataset()
    
    # 加载模型和tokenizer
    print(f"\n加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA配置
    print("配置 LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=16,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False
    )
    
    # 应用LoRA配置
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    print("\n可训练参数统计:")
    model.print_trainable_parameters()
    
    # 创建数据集
    print(f"\n创建数据集，共 {len(train_data)} 条样本")
    dataset = MedicalTermDataset(train_data, tokenizer, max_length)
    
    # 训练参数配置
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        logging_steps=10,
        save_steps=50,
        fp16=True,
        report_to="none",
        logging_dir="./logs",
        overwrite_output_dir=True
    )
    
    # 创建训练器
    print("\n创建训练器...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    # 开始训练
    print("\n开始 LoRA 微调训练...")
    trainer.train()
    
    # 保存模型
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"\nLoRA 权重已保存至: {output_dir}")
    
    # 保存数据集信息
    dataset_info = {
        "dataset_size": len(train_data),
        "model_name": model_name,
        "lora_rank": lora_rank,
        "epochs": epochs,
        "learning_rate": learning_rate
    }
    with open(os.path.join(output_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print("\n训练完成！")
    print("=" * 60)
    
    return model

def test_inference(model_path="./lora_medical_terms"):
    """测试微调后的模型"""
    from peft import PeftModel
    
    print("\n测试医疗术语解释模型...")
    
    model_name = "./models/Qwen_Qwen2.5-3B-Instruct/Qwen/Qwen2___5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载LoRA权重
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    # 测试术语
    test_terms = ["高血压", "糖尿病", "CT扫描", "心电图"]
    
    print("\n测试结果：")
    for term in test_terms:
        prompt = f"解释医疗术语：{term}\n解释："
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "解释：" in response:
            explanation = response.split("解释：")[1].strip()
        else:
            explanation = response.strip()
        
        print(f"\n术语: {term}")
        print(f"解释: {explanation}")

if __name__ == "__main__":
    # 执行训练
    train_lora_medical_terms()
    
    # 测试模型
    test_inference()