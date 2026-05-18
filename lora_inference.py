import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_lora_model(base_model_name="THUDM/glm-4-6b-chat", lora_path="./lora_output"):
    """加载带有 LoRA 权重的模型"""
    print(f"加载基础模型: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"加载 LoRA 权重: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    
    print("模型加载完成")
    return model, tokenizer

def generate_response(model, tokenizer, question, max_length=512):
    """生成回答"""
    prompt = f"问：{question}\n答："
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "答：" in response:
        answer = response.split("答：")[1].strip()
    else:
        answer = response.strip()
    
    return answer

if __name__ == "__main__":
    model, tokenizer = load_lora_model()
    
    print("\n医疗智能助手（LoRA微调版）")
    print("输入 'exit' 退出\n")
    
    while True:
        question = input("请输入问题：")
        if question.lower() == "exit":
            print("再见！")
            break
        
        try:
            answer = generate_response(model, tokenizer, question)
            print(f"回答：{answer}\n")
        except Exception as e:
            print(f"生成回答时出错: {e}\n")