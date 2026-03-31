from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rich.console import Console
from rich.markdown import Markdown
from tools import tools
from document_parser import DocumentParser
from vector_store import VectorStoreManager

# 加载环境变量
load_dotenv()

app = Flask(__name__)
CORS(app)  # 启用CORS

class MedicalReportAgent:
    def __init__(self, api_key=None, model="gpt-3.5-turbo", temperature=0.7, base_url=None):
        # 初始化LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url
        )
        
        # 初始化文档解析器
        self.parser = DocumentParser()
        
        # 初始化向量存储管理器
        self.vector_store = VectorStoreManager(api_key=api_key, base_url=base_url)
        
        # 创建工具描述
        tools_description = "\n".join([
            f"- {tool.name}: {tool.description}" 
            for tool in tools
        ])
        
        # 创建提示模板
        self.prompt_template = ChatPromptTemplate.from_template(
            f"""你是一个专业的医疗行业智能助手，专注于分析医疗研报和文档，并回答相关专业问题。

            对话历史：
            {{history}}

            用户问题：
            {{input}}
            
            文档检索结果：
            {{document_results}}
            
            可用工具：
            {tools_description}
            
            请根据文档内容和你的专业知识，提供详细、准确的回答。
            重要要求：
            1. 请使用纯文本格式回答，不要使用 markdown 语法（如 #、**、* 等）。
            2. 请将回答分成多个段落，每个段落集中表达一个主题，使重点清晰。
            3. 对于重要信息，可以适当使用换行来突出显示。
            
            如果用户的问题需要使用工具，请按照以下格式调用工具：
            工具名称: 参数
            
            例如：
            - 计算器: 2 + 2 * 3
            - 天气查询: 北京
            - 时间查询: (无参数)
            """
        )
        
        # 输出解析器
        self.output_parser = StrOutputParser()
        
        # 构建链
        self.chain = self.prompt_template | self.llm | self.output_parser
        
        # 对话历史
        self.history = []
    
    def run(self, user_input):
        """运行agent，处理用户输入"""
        # 构建上下文
        history_str = "\n".join([f"用户: {h[0]}\n助手: {h[1]}" for h in self.history])
        
        # 检索相关文档
        document_results = self.vector_store.search(user_input)
        
        # 调用链
        response = self.chain.invoke({
            "history": history_str,
            "input": user_input,
            "document_results": document_results
        })
        
        # 检查是否需要调用工具
        tool_result = self._check_tool_call(response)
        
        if tool_result:
            # 更新对话历史
            self.history.append((user_input, response))
            # 添加工具结果到历史
            self.history.append(("工具调用", tool_result))
            return tool_result
        else:
            # 更新对话历史
            self.history.append((user_input, response))
            return response
    
    def _check_tool_call(self, response):
        """检查响应中是否包含工具调用"""
        for tool in tools:
            if tool.name in response:
                # 提取工具参数
                try:
                    # 简单的参数提取逻辑
                    if ":" in response:
                        parts = response.split(":")
                        if len(parts) > 1:
                            param = parts[1].strip()
                            return tool.func(param)
                except Exception as e:
                    return f"工具调用错误: {str(e)}"
        return None
    
    def clear_history(self):
        """清空对话历史"""
        self.history = []
    
    def add_document(self, file_path):
        """添加文档到系统"""
        # 解析文档
        text = self.parser.parse_document(file_path)
        
        # 检查解析是否成功
        if "错误" in text:
            return text
        
        # 添加到向量存储
        result = self.vector_store.add_document(text, metadata={"file_path": file_path})
        return result
    
    def clear_documents(self):
        """清空所有文档"""
        return self.vector_store.clear()

# 初始化agent
API_KEY = "9d4fb1398d2a4446a8314c039467aaef.V0BpKtwTR4TLIhHP"  # 替换为你的实际API Key
MODEL = "glm-4.7-flash"  # 可以选择其他模型，如"gpt-4"
TEMPERATURE = 0.7  # 控制输出的随机性，0.0-1.0之间
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"  # 只包含基础API端点

agent = MedicalReportAgent(api_key=API_KEY, model=MODEL, temperature=TEMPERATURE, base_url=BASE_URL)

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
        # 保存文件到临时目录
        temp_dir = 'temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)
        
        # 添加文档到系统
        result = agent.add_document(file_path)
        
        # 删除临时文件
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
