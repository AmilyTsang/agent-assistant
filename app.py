from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import torch
from dotenv import load_dotenv

# 导入LoRA相关库（可选导入）
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    from langchain.chat_models import ChatOpenAI
except ImportError:
    from langchain_openai import ChatOpenAI

try:
    from langchain.prompts import ChatPromptTemplate
except ImportError:
    from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain.output_parsers import StrOutputParser
except ImportError:
    from langchain_core.output_parsers import StrOutputParser

# 导入Agent相关模块
try:
    from langchain.agents import initialize_agent, AgentType
    from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
    HAS_AGENT = True
except ImportError:
    HAS_AGENT = False

from tools import tools
from document_parser import DocumentParser
from vector_store import VectorStoreManager

# 加载环境变量
load_dotenv()

app = Flask(__name__)
CORS(app)  # 启用CORS

class MedicalReportAgent:
    def __init__(self, api_key=None, model="gpt-3.5-turbo", temperature=0.7, 
                 base_url=None, use_local_model=False, lora_path=None,
                 max_history_length=20, enable_planning=True):
        # 初始化文档解析器
        self.parser = DocumentParser()
        
        # 初始化向量存储管理器
        self.vector_store = VectorStoreManager(api_key=api_key, base_url=base_url)
        
        # 配置参数
        self.max_history_length = max_history_length
        self.enable_planning = enable_planning
        self.max_retries = 3
        
        # 对话历史
        self.history = []
        self.history_summary = ""
        
        # 选择模型类型
        self.use_local_model = use_local_model
        
        if use_local_model and lora_path and HAS_PEFT:
            # 使用本地LoRA微调模型
            self.llm = self._load_local_lora_model(lora_path)
            print(f"已加载本地LoRA模型: {lora_path}")
            self.enable_planning = False  # 本地模型暂不支持AgentExecutor
        else:
            # 使用API调用方式
            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key,
                base_url=base_url
            )
            print(f"已加载API模型: {model}")
            
            # 初始化AgentExecutor（具备规划能力）
            if HAS_AGENT and self.enable_planning:
                self._init_agent_executor()
        
    def _load_local_lora_model(self, lora_path):
        """加载本地LoRA微调模型"""
        base_model_name = "THUDM/glm-4-6b-chat"
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载LoRA权重
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()
        
        return (model, tokenizer)
    
    def _init_agent_executor(self):
        """初始化具备规划能力的AgentExecutor"""
        # 创建记忆系统
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_len=self.max_history_length
        )
        
        # 创建AgentExecutor
        self.agent_executor = initialize_agent(
            tools,
            self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        print("AgentExecutor 初始化完成，支持自主规划和多工具协作")
    
    def _summarize_history(self):
        """对对话历史进行摘要压缩"""
        if len(self.history) <= 5:
            return ""
        
        # 构建摘要提示
        history_str = "\n".join([f"用户: {h[0]}\n助手: {h[1]}" for h in self.history[:-5]])
        
        summary_prompt = f"""请对以下对话历史进行简要总结，提取关键信息：

{history_str}

总结："""
        
        if self.use_local_model and isinstance(self.llm, tuple):
            model, tokenizer = self.llm
            inputs = tokenizer(summary_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=256,
                    temperature=0.3,
                    do_sample=False
                )
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "总结：" in summary:
                summary = summary.split("总结：")[1].strip()
        else:
            summary = self.llm.predict(summary_prompt)
        
        return summary
    
    def _truncate_history(self):
        """截断历史，保持最大长度限制"""
        if len(self.history) > self.max_history_length:
            # 先进行摘要压缩
            self.history_summary = self._summarize_history()
            # 保留最近的历史
            self.history = self.history[-self.max_history_length // 2:]
    
    def _check_tool_call(self, response):
        """检查响应中是否包含工具调用"""
        for tool in tools:
            if tool.name in response:
                try:
                    if ":" in response:
                        parts = response.split(":")
                        if len(parts) > 1:
                            param = parts[1].strip()
                            return tool.func(param)
                except Exception as e:
                    return f"工具调用错误: {str(e)}"
        return None
    
    def _reflect_and_adjust(self, user_input, error):
        """反思错误并调整策略"""
        reflect_prompt = f"""
用户问题：{user_input}
执行错误：{error}

请分析错误原因，并给出调整后的提问方式或解决方案。
"""
        
        if self.use_local_model and isinstance(self.llm, tuple):
            model, tokenizer = self.llm
            inputs = tokenizer(reflect_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    temperature=0.5,
                    do_sample=True
                )
            reflection = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            reflection = self.llm.predict(reflect_prompt)
        
        return reflection
    
    def run(self, user_input):
        """运行agent，处理用户输入（带规划能力）"""
        # 检查是否启用规划能力
        if HAS_AGENT and self.enable_planning and not self.use_local_model:
            return self._run_with_planning(user_input)
        else:
            return self._run_basic(user_input)
    
    def _run_with_planning(self, user_input):
        """使用AgentExecutor进行规划执行"""
        # 更新历史
        self.history.append((user_input, ""))
        
        # 截断历史
        self._truncate_history()
        
        try:
            # 检索相关文档
            document_results = self.vector_store.search(user_input)
            
            # 将文档检索结果添加到提示中
            enhanced_input = f"""
用户问题：{user_input}

参考文档：
{document_results}

请根据文档内容和你的专业知识回答问题。
"""
            
            # 使用AgentExecutor执行
            for attempt in range(self.max_retries):
                try:
                    response = self.agent_executor.run(enhanced_input)
                    
                    # 检查是否需要进一步处理
                    if self._needs_follow_up(response):
                        user_input = self._generate_follow_up(response)
                        continue
                    
                    # 更新历史
                    self.history[-1] = (user_input, response)
                    return response
                    
                except Exception as e:
                    # 反思并调整策略
                    if attempt < self.max_retries - 1:
                        user_input = self._reflect_and_adjust(user_input, str(e))
                    else:
                        return f"多次尝试后仍无法完成任务，请尝试其他方式。错误信息：{str(e)}"
            
        except Exception as e:
            return f"执行错误: {str(e)}"
    
    def _run_basic(self, user_input):
        """基础执行模式（无规划能力）"""
        # 截断历史
        self._truncate_history()
        
        # 构建上下文
        history_str = ""
        if self.history_summary:
            history_str = f"对话摘要：{self.history_summary}\n\n"
        history_str += "\n".join([f"用户: {h[0]}\n助手: {h[1]}" for h in self.history])
        
        # 检索相关文档
        document_results = self.vector_store.search(user_input)
        
        # 根据模型类型选择不同的推理方式
        if self.use_local_model and isinstance(self.llm, tuple):
            response = self._local_inference(user_input, document_results, history_str)
        else:
            # 构建提示模板
            tools_description = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
            
            prompt = ChatPromptTemplate.from_template(
                f"""你是一个专业的医疗行业智能助手。

对话历史：
{{history}}

用户问题：
{{input}}

文档检索结果：
{{document_results}}

可用工具：
{tools_description}

请提供详细、准确的回答。如果需要使用工具，请按格式输出：工具名称: 参数
"""
            )
            chain = prompt | self.llm | StrOutputParser()
            
            response = chain.invoke({
                "history": history_str,
                "input": user_input,
                "document_results": document_results
            })
        
        # 检查是否需要调用工具
        tool_result = self._check_tool_call(response)
        
        if tool_result:
            # 更新对话历史
            self.history.append((user_input, response))
            self.history.append(("工具调用", tool_result))
            return tool_result
        else:
            # 更新对话历史
            self.history.append((user_input, response))
            return response
    
    def _local_inference(self, user_input, document_results, history_str):
        """本地模型推理"""
        model, tokenizer = self.llm
        
        tools_description = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        
        prompt = f"""你是一个专业的医疗行业智能助手，专注于分析医疗研报和文档，并回答相关专业问题。

对话历史：
{history_str}

用户问题：
{user_input}

文档检索结果：
{document_results}

可用工具：
{tools_description}

请提供详细、准确的回答。如果需要使用工具，请按格式输出：工具名称: 参数
"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    
    def _needs_follow_up(self, response):
        """判断是否需要后续处理"""
        follow_up_keywords = ["需要更多信息", "请提供", "请补充", "需要确认"]
        return any(keyword in response for keyword in follow_up_keywords)
    
    def _generate_follow_up(self, response):
        """生成后续问题"""
        return f"请提供更多信息以便继续处理：{response}"
    
    def clear_history(self):
        """清空对话历史"""
        self.history = []
        self.history_summary = ""
        if HAS_AGENT and hasattr(self, 'memory'):
            self.memory.clear()
    
    def add_document(self, file_path):
        """添加文档到系统"""
        text = self.parser.parse_document(file_path)
        if "错误" in text:
            return text
        result = self.vector_store.add_document(text, metadata={"file_path": file_path})
        return result
    
    def clear_documents(self):
        """清空所有文档"""
        return self.vector_store.clear()

# 配置选项
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
LORA_PATH = os.getenv("LORA_PATH", "./lora_output")
ENABLE_PLANNING = os.getenv("ENABLE_PLANNING", "true").lower() == "true"

# 初始化agent
API_KEY = os.getenv("OPENAI_API_KEY", "a73fa9b9137441ea9c835949a7c19c5e.an4Y8vj2Fs5QO1He")
MODEL = os.getenv("MODEL", "glm-5v-turbo")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
BASE_URL = os.getenv("BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")

# 验证API Key（仅在使用API模式时）
if not USE_LOCAL_MODEL and (not API_KEY or API_KEY == "your-openai-api-key"):
    raise ValueError("请在.env文件中设置OPENAI_API_KEY环境变量")

agent = MedicalReportAgent(
    api_key=API_KEY, 
    model=MODEL, 
    temperature=TEMPERATURE, 
    base_url=BASE_URL,
    use_local_model=USE_LOCAL_MODEL,
    lora_path=LORA_PATH,
    enable_planning=ENABLE_PLANNING
)

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message')
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        response = agent.run(user_input)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear():
    agent.clear_history()
    return jsonify({"message": "History cleared"})

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        temp_dir = 'temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)
        
        result = agent.add_document(file_path)
        
        os.remove(file_path)
        
        return jsonify({"message": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear_docs', methods=['POST'])
def clear_docs():
    try:
        result = agent.clear_documents()
        return jsonify({"message": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "service": "medical-report-agent",
        "model": MODEL,
        "use_local_model": USE_LOCAL_MODEL,
        "enable_planning": ENABLE_PLANNING,
        "version": "1.0.0"
    })

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("FLASK_ENV", "production").lower() == "development"
    app.run(host='0.0.0.0', port=port, debug=debug_mode)