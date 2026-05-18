"""
医疗术语解释模型推理脚本
使用LoRA微调后的模型解释医疗报告中的术语
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class MedicalTermExplainer:
    """医疗术语解释器"""
    
    def __init__(self, base_model_name="THUDM/glm-4-6b-chat", lora_path="./lora_medical_terms"):
        """
        初始化医疗术语解释器
        
        Args:
            base_model_name: 基础模型名称
            lora_path: LoRA权重路径
        """
        print(f"加载基础模型: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"加载LoRA权重: {lora_path}")
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()
        
        print("医疗术语解释器初始化完成")
    
    def explain_term(self, term, max_length=512, temperature=0.7):
        """
        解释医疗术语
        
        Args:
            term: 要解释的医疗术语
            max_length: 生成的最大长度
            temperature: 生成温度
            
        Returns:
            术语解释文本
        """
        # 构建提示
        prompt = f"解释医疗术语：{term}\n解释："
        
        # 编码
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True
            )
        
        # 解码
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取解释部分
        if "解释：" in response:
            explanation = response.split("解释：")[1].strip()
        else:
            explanation = response.strip()
        
        return explanation
    
    def batch_explain(self, terms):
        """
        批量解释多个医疗术语
        
        Args:
            terms: 术语列表
            
        Returns:
            术语解释字典
        """
        results = {}
        for term in terms:
            try:
                results[term] = self.explain_term(term)
            except Exception as e:
                results[term] = f"解释失败: {str(e)}"
        return results

if __name__ == "__main__":
    # 创建解释器
    explainer = MedicalTermExplainer()
    
    print("\n医疗术语解释器")
    print("输入 'exit' 退出\n")
    
    while True:
        term = input("请输入医疗术语：")
        if term.lower() == "exit":
            print("再见！")
            break
        
        try:
            explanation = explainer.explain_term(term)
            print(f"\n术语解释：")
            print(explanation)
            print("\n" + "="*60 + "\n")
        except Exception as e:
            print(f"\n解释失败: {e}\n")