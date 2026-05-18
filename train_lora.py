import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import os

class MedicalDataset(Dataset):
    """医疗领域微调数据集"""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        instruction = item["instruction"]
        input_text = item["input"]
        output_text = item["output"]
        
        prompt = f"问：{instruction}"
        if input_text:
            prompt += f"\n上下文：{input_text}"
        prompt += f"\n答："
        
        full_text = prompt + output_text
        
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        prompt_length = len(self.tokenizer.encode(prompt))
        
        labels = encoding["input_ids"].clone()
        labels[0, :prompt_length] = -100
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": labels.flatten()
        }

def train_lora():
    """执行 LoRA 微调训练"""
    
    model_name = "THUDM/glm-4-6b-chat"
    data_path = "data/medical_finetune.json"
    output_dir = "./lora_output"
    lora_rank = 8
    batch_size = 2
    epochs = 3
    learning_rate = 3e-4
    
    print(f"加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
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
    model = get_peft_model(model, lora_config)
    
    print("可训练参数:")
    model.print_trainable_parameters()
    
    print(f"加载数据集: {data_path}")
    dataset = MedicalDataset(data_path, tokenizer)
    
    print("配置训练参数...")
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
        logging_dir="./logs"
    )
    
    print("创建训练器...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    
    print("开始 LoRA 微调训练...")
    trainer.train()
    
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"LoRA 权重已保存至 {output_dir}")
    
    return model

if __name__ == "__main__":
    train_lora()