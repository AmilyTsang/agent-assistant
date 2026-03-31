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

# 初始化控制台
console = Console()

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

def main():
    # 在这里设置API相关参数
    API_KEY = "9d4fb1398d2a4446a8314c039467aaef.V0BpKtwTR4TLIhHP"  # 替换为你的实际API Key
    MODEL = "glm-5"  # 可以选择其他模型，如"gpt-4"
    TEMPERATURE = 0.7  # 控制输出的随机性，0.0-1.0之间
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4/" # 设置API的基础URL，如果使用代理服务，例如："https://api.openai.com/v1"
    
    console.print("[bold cyan]=== 医疗行业研报智能助手 ===[/bold cyan]")
    console.print("输入你的问题，输入 'exit' 退出，输入 'clear' 清空历史\n")
    console.print("[bold yellow]可用命令:[/bold yellow]\n")
    console.print("- add_doc <文件路径>: 添加医疗研报文档\n")
    console.print("- clear_docs: 清空所有文档\n")
    console.print("- clear: 清空对话历史\n")
    console.print("- exit: 退出程序\n")
    
    # 使用main函数中设置的参数初始化agent
    agent = MedicalReportAgent(api_key=API_KEY, model=MODEL, temperature=TEMPERATURE, base_url=BASE_URL)
    
    while True:
        user_input = console.input("[bold green]你: [/bold green]")
        
        if user_input.lower() == 'exit':
            console.print("[bold blue]助手:[/bold blue] 再见！")
            break
        
        if user_input.lower() == 'clear':
            agent.clear_history()
            console.print("[bold blue]助手:[/bold blue] 历史已清空")
            continue
        
        if user_input.lower() == 'clear_docs':
            result = agent.clear_documents()
            console.print(f"[bold blue]助手:[/bold blue] {result}")
            continue
        
        if user_input.lower().startswith('add_doc'):
            # 提取文件路径
            parts = user_input.split(' ', 1)
            if len(parts) > 1:
                file_path = parts[1].strip()
                result = agent.add_document(file_path)
                console.print(f"[bold blue]助手:[/bold blue] {result}")
            else:
                console.print("[bold blue]助手:[/bold blue] 请提供文件路径")
            continue
        
        try:
            response = agent.run(user_input)
            console.print("[bold blue]助手:[/bold blue]")
            console.print(Markdown(response))
            console.print()
        except Exception as e:
            console.print(f"[bold red]错误:[/bold red] {e}")
            console.print("请检查OpenAI API Key是否正确配置")

if __name__ == "__main__":
    main()
