from langchain_core.tools import Tool
from pydantic import BaseModel, Field
import requests
import json
from datetime import datetime

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
    """解释医疗术语的含义"""
    try:
        # 这里可以集成专业的医疗术语API，现在使用模拟数据
        medical_terms = {
            "高血压": "高血压是指动脉血压持续升高，收缩压≥140mmHg和/或舒张压≥90mmHg。长期高血压可导致心脏病、脑卒中等并发症。",
            "糖尿病": "糖尿病是一种代谢性疾病，特征是血糖水平持续升高。主要分为1型和2型，需要通过饮食控制、运动和药物治疗。",
            "冠心病": "冠心病是由于冠状动脉粥样硬化导致心肌缺血缺氧的疾病，常见症状为心绞痛、心肌梗死等。",
            "脑卒中": "脑卒中又称中风，是由于脑部血管阻塞或破裂导致脑组织损伤的疾病，可分为缺血性和出血性两种。",
            "癌症": "癌症是由细胞异常增生形成的恶性肿瘤，可发生在身体的任何部位，早期发现和治疗至关重要。"
        }
        
        if term in medical_terms:
            return f"{term}：{medical_terms[term]}"
        else:
            return f"未找到{term}的详细解释，建议咨询专业医生。"
    except Exception as e:
        return f"解释医疗术语时出错: {str(e)}"

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
