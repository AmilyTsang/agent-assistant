try:
    from langchain.tools import Tool
except ImportError:
    from langchain_core.tools import Tool

try:
    from pydantic.v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field
import requests
import json
from datetime import datetime
import os

# 尝试导入LoRA模型相关库
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

# 全局医疗术语解释器
medical_explainer = None

class MedicalTermExplainer:
    """医疗术语解释器（基于LoRA微调模型）"""
    
    def __init__(self, base_model_name="THUDM/glm-4-6b-chat", lora_path="./lora_medical_terms"):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()
    
    def explain(self, term):
        prompt = f"解释医疗术语：{term}\n解释："
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "解释：" in response:
            return response.split("解释：")[1].strip()
        return response.strip()

def init_medical_explainer():
    """初始化医疗术语解释器"""
    global medical_explainer
    if HAS_PEFT and os.path.exists("./lora_medical_terms"):
        try:
            medical_explainer = MedicalTermExplainer()
            print("医疗术语解释器（LoRA模型）初始化成功")
        except Exception as e:
            print(f"初始化LoRA模型失败，使用内置数据: {e}")
    else:
        print("LoRA模型不可用，使用内置医疗术语数据")

# 计算器工具
class CalculatorInput(BaseModel):
    expression: str = Field(description="数学表达式，例如：2 + 2 * 3")

def calculator(expression: str) -> str:
    """计算数学表达式并返回结果"""
    try:
        # 简单的表达式计算
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 天气查询工具
class WeatherInput(BaseModel):
    city: str = Field(description="城市名称，例如：北京")

def get_weather(city: str) -> str:
    """查询指定城市的天气信息"""
    try:
        # 使用OpenWeatherMap API
        api_key = "YOUR_OPENWEATHER_API_KEY"  # 替换为你的API Key
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=zh_cn"
        response = requests.get(url)
        data = response.json()
        
        if data.get("cod") == 200:
            weather = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            return f"{city}的天气: {weather}，温度: {temp}°C，湿度: {humidity}%"
        else:
            return f"无法查询{city}的天气信息"
    except Exception as e:
        return f"查询天气时出错: {str(e)}"

# 时间查询工具
def get_current_time() -> str:
    """获取当前时间"""
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"

# 医疗术语解释工具
class MedicalTermInput(BaseModel):
    term: str = Field(description="医疗术语，例如：高血压")

def explain_medical_term(term: str) -> str:
    """解释医疗术语的含义（优先使用LoRA模型）"""
    global medical_explainer
    
    # 尝试使用LoRA模型
    if medical_explainer:
        try:
            result = medical_explainer.explain(term)
            return f"{term}：{result}"
        except Exception as e:
            print(f"LoRA模型解释失败，降级到内置数据: {e}")
    
    # 降级到内置数据
    medical_terms = {
        "高血压": "高血压是指动脉血压持续升高，收缩压≥140mmHg和/或舒张压≥90mmHg。长期高血压可导致心脏病、脑卒中等并发症。",
        "糖尿病": "糖尿病是一种代谢性疾病，特征是血糖水平持续升高。主要分为1型和2型，需要通过饮食控制、运动和药物治疗。",
        "冠心病": "冠心病是由于冠状动脉粥样硬化导致心肌缺血缺氧的疾病，常见症状为心绞痛、心肌梗死等。",
        "脑卒中": "脑卒中又称中风，是由于脑部血管阻塞或破裂导致脑组织损伤的疾病，可分为缺血性和出血性两种。",
        "癌症": "癌症是由细胞异常增生形成的恶性肿瘤，可发生在身体的任何部位，早期发现和治疗至关重要。",
        "肺炎": "肺炎是肺部的感染性疾病，通常由细菌、病毒或真菌引起，常见症状包括发热、咳嗽、呼吸困难等。",
        "哮喘": "哮喘是一种慢性气道炎症性疾病，特征是气道狭窄和呼吸困难，常由过敏原、感染或运动诱发。",
        "关节炎": "关节的炎症性疾病，常见症状包括关节疼痛、肿胀和僵硬，严重时可导致关节畸形。",
        "CT扫描": "CT扫描即计算机断层扫描，是一种利用X射线穿透人体并通过计算机处理形成断层图像的医学检查技术，可用于诊断多种疾病。",
        "MRI": "MRI即磁共振成像，是利用磁场和无线电波生成人体内部结构详细图像的检查方法，无辐射，对软组织成像效果好。",
        "心电图": "心电图是记录心脏电活动的检查方法，通过在体表放置电极来检测心脏的节律和心肌缺血情况，常用于诊断心律失常和心肌梗死。",
        "血常规": "血常规是通过检测血液中的红细胞、白细胞、血小板等指标来评估身体基本健康状况的检查，可发现感染、贫血等问题。"
    }
    
    if term in medical_terms:
        return f"{term}：{medical_terms[term]}"
    else:
        return f"未找到{term}的详细解释，建议咨询专业医生。"

# 药品信息查询工具
class DrugInfoInput(BaseModel):
    drug_name: str = Field(description="药品名称，例如：阿司匹林")

def get_drug_info(drug_name: str) -> str:
    """查询药品的基本信息"""
    try:
        # 这里可以集成专业的药品信息API，现在使用模拟数据
        drugs = {
            "阿司匹林": "阿司匹林是一种非甾体抗炎药，具有解热、镇痛、抗炎和抗血小板聚集作用。常用于缓解疼痛、降低体温、预防心脑血管疾病等。",
            "布洛芬": "布洛芬是一种非甾体抗炎药，具有解热、镇痛、抗炎作用。常用于缓解轻至中度疼痛，如头痛、关节痛、牙痛等。",
            "对乙酰氨基酚": "对乙酰氨基酚是一种解热镇痛药，主要用于缓解疼痛和降低体温，对炎症的作用较弱。",
            "青霉素": "青霉素是一种抗生素，用于治疗细菌感染，如肺炎、扁桃体炎、中耳炎等。",
            "胰岛素": "胰岛素是一种激素，用于治疗糖尿病，帮助调节血糖水平。"
        }
        
        if drug_name in drugs:
            return f"{drug_name}：{drugs[drug_name]}"
        else:
            return f"未找到{drug_name}的详细信息，建议咨询专业医生或药师。"
    except Exception as e:
        return f"查询药品信息时出错: {str(e)}"

# 定义工具
tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="用于计算数学表达式",
        args_schema=CalculatorInput
    ),
    Tool(
        name="Weather",
        func=get_weather,
        description="用于查询城市天气",
        args_schema=WeatherInput
    ),
    Tool(
        name="CurrentTime",
        func=get_current_time,
        description="获取当前时间"
    ),
    Tool(
        name="MedicalTerm",
        func=explain_medical_term,
        description="用于解释医疗术语的含义",
        args_schema=MedicalTermInput
    ),
    Tool(
        name="DrugInfo",
        func=get_drug_info,
        description="用于查询药品的基本信息",
        args_schema=DrugInfoInput
    )
]

# 初始化医疗术语解释器
init_medical_explainer()
