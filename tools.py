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
    )
]
