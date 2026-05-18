from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import torch
from datetime import datetime
from dotenv import load_dotenv

# ========================
# 安全导入（Qwen2 专用）
# ========================
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, PeftConfig
    HAS_PEFT = True
    print("✅ Transformers & PEFT 库已加载")
except ImportError as e:
    HAS_PEFT = False
    print(f"⚠️ 缺少必要库: {e}")

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "glm-5v-turbo")
BASE_URL = os.getenv("BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
LORA_PATH = os.getenv("LORA_PATH", "./lora_medical_sampled")
PORT = int(os.getenv("PORT", 5000))

# 你的本地 Qwen 模型路径
LOCAL_QWEN_PATH = r"C:\Users\15637\Desktop\个人项目总结\算法所有的项目上传版\agent-assistant\models\Qwen\Qwen2___5-3B-Instruct"

app = Flask(__name__)
CORS(app)

# ==================================================
# 内置医疗术语数据库（回退方案）
# ==================================================
MEDICAL_TERMS = {
    "高血压": "高血压是指以体循环动脉血压（收缩压≥140mmHg和/或舒张压≥90mmHg）为主要特征，可伴有心、脑、肾等器官功能损害的临床综合征。",
    "糖尿病": "糖尿病是一组以高血糖为特征的代谢性疾病。长期高血糖会导致眼、肾、心脏、血管、神经的慢性损害。",
    "阿司匹林": "阿司匹林具有解热镇痛、抗炎、抗血小板聚集作用。常用于预防心脑血管疾病。",
    "血常规": "血常规是最基本的血液检验，主要指标包括白细胞计数、红细胞计数、血红蛋白浓度和血小板计数。",
    "肿瘤标志物": "肿瘤标志物是由肿瘤细胞产生或机体对肿瘤反应产生的物质，用于辅助诊断癌症。",
    "冠心病": "冠心病是冠状动脉粥样硬化性心脏病的简称，由于冠状动脉狭窄导致心肌缺血缺氧。",
    "心肌梗死": "心肌梗死是冠状动脉急性持续缺血缺氧引起的心肌坏死，临床表现为剧烈胸痛。",
    "脑卒中": "脑卒中是急性脑血管疾病，分为缺血性和出血性卒中，主要危险因素包括高血压和糖尿病。",
    "肺炎": "肺炎是终末气道、肺泡和肺间质的炎症，常见病原体包括细菌、病毒、真菌。",
    "哮喘": "哮喘是一种慢性气道炎症性疾病，特征为可逆性气流受限，典型症状包括喘息和气急。"
}

# ==================================================
# 本地 LoRA 模型（Qwen2 专用）
# ==================================================
class LocalLoraModel:
    def __init__(self, lora_path: str, base_model_path: str):
        self.lora_path = lora_path
        self.base_model_path = base_model_path
        self.model = None
        self.tokenizer = None
        self.loaded = False
        
        if HAS_PEFT:
            self._load()
        else:
            print("⚠️ 缺少 PEFT 库，无法加载 LoRA 模型")
    
    def _load(self):
        if not os.path.exists(self.lora_path):
            print(f"⚠️ LoRA 路径不存在: {self.lora_path}")
            return
        
        if not os.path.exists(self.base_model_path):
            print(f"⚠️ 基础模型路径不存在: {self.base_model_path}")
            return
        
        print(f"🔄 加载基础模型: {self.base_model_path}")
        print(f"🔄 加载 LoRA 权重: {self.lora_path}")
        
        try:
            # 1. 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # 2. 加载基础模型
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True
            )
            
            # 3. 尝试加载 LoRA（多种方式）
            try:
                # 方式1：标准加载
                model = PeftModel.from_pretrained(model, self.lora_path)
            except Exception as e1:
                print(f"⚠️ 标准加载失败: {e1}")
                try:
                    # 方式2：尝试加载配置并修复
                    config = PeftConfig.from_pretrained(self.lora_path)
                    # 移除可能导致问题的参数
                    config_dict = config.to_dict()
                    config_dict.pop('alora_invocation_tokens', None)
                    config_dict.pop('alora_ranking_coeff', None)
                    config_dict.pop('megatron_core', None)
                    
                    # 尝试从字典重新创建配置
                    config = PeftConfig.from_dict(config_dict)
                    model = PeftModel.from_pretrained(model, self.lora_path, config=config)
                except Exception as e2:
                    print(f"⚠️ 配置修复加载失败: {e2}")
                    try:
                        # 方式3：跳过 embed_tokens 相关的错误
                        model = PeftModel.from_pretrained(
                            model, 
                            self.lora_path,
                            ignore_mismatched_sizes=True  # 忽略尺寸不匹配
                        )
                    except Exception as e3:
                        print(f"⚠️ 忽略尺寸不匹配加载失败: {e3}")
                        raise Exception(f"所有加载方式均失败: {e1}, {e2}, {e3}")
            
            model.eval()
            self.model = model
            self.loaded = True
            print("✅ LoRA 模型加载成功")
            
        except Exception as e:
            print(f"❌ LoRA 模型加载失败: {e}")
            self.loaded = False
    
    def infer(self, prompt: str) -> str:
        if not self.loaded:
            return None
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=TEMPERATURE,
                    do_sample=True,
                    top_p=0.9
                )
            return self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
        except Exception as e:
            print(f"⚠️ LoRA 推理失败: {e}")
            return None

# ==================================================
# API 模型
# ==================================================
class ApiModel:
    def __init__(self):
        self.available = False
        try:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=MODEL,
                temperature=TEMPERATURE,
                api_key=API_KEY,
                base_url=BASE_URL
            )
            self.available = True
            print("✅ API 模型初始化成功")
        except Exception as e:
            print(f"⚠️ API 模型初始化失败: {e}")
    
    def infer(self, prompt: str) -> str:
        if not self.available:
            return None
        try:
            return self.llm.predict(prompt)
        except Exception as e:
            print(f"⚠️ API 调用失败: {e}")
            return None

# ==================================================
# 医疗助手（智能回退）
# ==================================================
class MedicalAssistant:
    def __init__(self):
        print("\n" + "="*50)
        print("🏥 初始化医疗助手...")
        
        # 初始化本地 LoRA 模型
        self.local_model = LocalLoraModel(LORA_PATH, LOCAL_QWEN_PATH)
        
        # 初始化 API 模型
        self.api_model = ApiModel()
        
        # 当前使用的模型
        self.current_model = "fallback"
        
        print("="*50 + "\n")
    
    def answer(self, question: str) -> str:
        """回答问题，按优先级尝试不同模型"""
        
        # 1. 尝试本地 LoRA 模型
        if self.local_model.loaded:
            print("🏠 使用本地 LoRA 模型...")
            response = self.local_model.infer(question)
            if response:
                self.current_model = "local"
                return response
        
        # 2. 尝试 API 模型
        if self.api_model.available:
            print("☁️ 使用 API 模型...")
            response = self.api_model.infer(question)
            if response:
                self.current_model = "api"
                return response
        
        # 3. 使用内置医疗术语（最终回退）
        print("📚 使用内置医疗术语...")
        self.current_model = "fallback"
        return self._fallback_answer(question)
    
    def _fallback_answer(self, question: str) -> str:
        """内置医疗术语回退"""
        question_lower = question.lower()
        
        # 精确匹配术语
        for term, definition in MEDICAL_TERMS.items():
            if term.lower() in question_lower:
                return f"📖 **{term}**\n\n{definition}"
        
        # 关键词匹配
        keywords = {
            "血压": "高血压",
            "血糖": "糖尿病",
            "吃药": "用药",
            "检查": "体检",
            "化验": "血常规"
        }
        
        for key, term in keywords.items():
            if key in question_lower and term in MEDICAL_TERMS:
                return f"📖 **{term}**\n\n{MEDICAL_TERMS[term]}"
        
        # 通用回答
        return """我是医疗研报智能助手，可以为您解答医疗相关问题。

📚 **我可以帮您：**
• 解释医疗术语（如高血压、糖尿病等）
• 说明检查项目的意义
• 介绍常见疾病的基本知识

💡 **请尝试询问：**
• "什么是高血压？"
• "糖尿病有什么症状？"
• "血常规检查包括哪些项目？"

⚠️ **温馨提示：** 我提供的信息仅供参考，具体诊疗请咨询专业医生。"""

# ==================================================
# Flask 服务
# ==================================================
assistant = MedicalAssistant()

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_input = data.get("message", "")
    
    if not user_input:
        return jsonify({"error": "请输入您的问题"}), 400
    
    try:
        response = assistant.answer(user_input)
        return jsonify({
            "response": response,
            "model": assistant.current_model,
            "time": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "response": "抱歉，我暂时无法回答这个问题。",
            "error": str(e)
        })

@app.route("/api/models", methods=["GET"])
def get_models():
    """获取可用模型状态"""
    return jsonify({
        "local": assistant.local_model.loaded,
        "api": assistant.api_model.available,
        "current": assistant.current_model
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "models": {
            "local": assistant.local_model.loaded,
            "api": assistant.api_model.available
        }
    })

if __name__ == "__main__":
    print("\n🚀 启动医疗研报智能助手...")
    print(f"🌐 访问地址: http://localhost:{PORT}")
    print(f"📁 LoRA 路径: {LORA_PATH}")
    print(f"🤖 基础模型: {LOCAL_QWEN_PATH}")
    print(f"🔑 API 可用: {assistant.api_model.available}")
    print(f"🏠 本地模型: {assistant.local_model.loaded}")
    print("\n" + "="*50 + "\n")
    
    app.run(host="0.0.0.0", port=PORT, debug=True)