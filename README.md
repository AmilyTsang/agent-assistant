# 医疗行业研报智能助手 (Medical Report Agent)

基于 LangChain 和大语言模型构建的智能医疗研报分析助手，能够自动解析医疗行业文档，支持专业问答、医疗知识查询、数据分析等功能。

## 项目功能

- **文档解析**：支持 PDF 和 Word 格式的医疗研报解析
- **向量检索**：基于 FAISS 实现文档向量化存储与语义相似性搜索
- **专业问答**：基于 RAG（检索增强生成）架构，结合文档内容和专业知识回答问题
- **医疗工具集**：内置计算器、天气查询、时间查询、医疗术语解释、药品信息查询等工具
- **智能体架构**：基于 LangChain Agent 的工具调用型智能体，支持多轮对话
- **Web 界面**：现代化的前端界面，支持文档上传、对话管理、报告生成
- **命令行模式**：支持命令行交互模式，方便调试和快速测试

## 技术栈

### 后端技术
| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.8+ | 开发语言 |
| Flask | 2.0+ | Web 框架 |
| LangChain | 0.3+ | 智能体框架 |
| FAISS | - | 向量数据库 |
| PyPDF2 | - | PDF 解析 |
| python-docx | - | Word 解析 |
| scikit-learn | - | TF-IDF 嵌入 |
| pydantic | <2.0 | 数据验证 |

### 前端技术
| 技术 | 用途 |
|------|------|
| HTML5 | 页面结构 |
| CSS3 | 样式设计 |
| JavaScript (ES6+) | 交互逻辑 |

### AI/ML 技术
| 技术 | 用途 |
|------|------|
| GLM-4 / GPT-3.5 | 大语言模型 |
| TF-IDF | 文本向量化 |
| FAISS | 向量相似度检索 |
| LangChain Agent | 工具调用型智能体 |

## 快速开始

### 1. 安装依赖

```bash
# 克隆项目后进入目录
cd agent-assistant

# 使用 pip 安装依赖
python -m pip install -r requirements.txt

# 或使用 conda 创建环境
conda create -n medical-agent python=3.9
conda activate medical-agent
python -m pip install -r requirements.txt
```

### 2. 配置环境变量

复制环境变量示例文件并配置：

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API Key：

```env
# 智谱 GLM API
OPENAI_API_KEY=your-api-key-here
MODEL=glm-4
BASE_URL=https://open.bigmodel.cn/api/paas/v4/
TEMPERATURE=0.7
```

### 3. 运行项目

```bash
# 启动 Web 服务
python app.py

# 或启动命令行模式
python main.py
```

### 4. 访问使用

- **Web 界面**：打开浏览器访问 `http://localhost:5000`
- **命令行模式**：在终端中进行交互式对话

## 使用方法

### Web 界面模式

1. **上传文档**：点击"选择文件"按钮，上传 PDF 或 Word 格式的医疗研报
2. **等待解析**：系统自动解析文档并进行向量化存储
3. **提问问题**：在输入框中输入关于医疗研报的问题
4. **获取回答**：智能助手结合文档内容和专业知识提供回答
5. **管理文档**：可以删除单个文档或清空所有文档
6. **清空历史**：点击"清空历史"按钮清除对话记录

### 命令行模式

```
=== 医疗行业研报智能助手 ===

可用命令:
- add_doc <文件路径>: 添加医疗研报文档
- clear_docs: 清空所有文档
- clear: 清空对话历史
- exit: 退出程序

你: add_doc ./医疗研报.pdf
你: 分析这篇研报的主要内容
助手: 根据文档内容，这篇研报主要讨论了...
```

## 项目结构

```
agent-assistant/
├── frontend/              # 前端文件
│   └── index.html        # Web 界面主页面
├── src/                   # 源代码目录（待扩展）
├── temp/                  # 临时文件存储
├── app.py                 # Flask Web 服务入口
├── main.py                # 命令行模式入口
├── document_parser.py     # 文档解析模块
├── vector_store.py        # 向量存储与检索模块
├── tools.py               # 工具定义模块
├── test_simple.py         # 基础功能测试脚本
├── test_agent.py          # 完整功能测试脚本
├── requirements.txt       # Python 依赖
├── .env                   # 环境变量配置
├── .env.example          # 环境变量示例
└── README.md             # 项目说明文档
```

## 核心模块说明

### MedicalReportAgent 类

智能体核心类，负责处理用户输入和生成回答。

| 方法 | 功能 |
|------|------|
| `__init__(api_key, model, temperature, base_url)` | 初始化 LLM、文档解析器、向量存储 |
| `run(user_input)` | 处理用户输入，检索文档并生成响应 |
| `add_document(file_path)` | 添加文档到系统并进行向量化 |
| `clear_documents()` | 清空所有已添加的文档 |
| `clear_history()` | 清空对话历史记录 |

### DocumentParser 类

文档解析器，支持 PDF 和 Word 格式。

| 方法 | 功能 |
|------|------|
| `parse_pdf(file_path)` | 解析 PDF 文件，提取文本内容 |
| `parse_docx(file_path)` | 解析 Word 文件，提取文本内容 |
| `parse_document(file_path)` | 根据文件扩展名自动选择解析方法 |

### VectorStoreManager 类

向量存储管理器，使用 FAISS 实现高效的相似性搜索。

| 方法 | 功能 |
|------|------|
| `add_document(text, metadata)` | 添加文档并进行向量化存储 |
| `search(query, k=3)` | 检索与查询最相关的 k 个文档片段 |
| `clear()` | 清空向量存储 |

### 工具模块 (tools.py)

内置的专业工具，用于扩展智能体能力。

| 工具名称 | 功能 | 示例 |
|---------|------|------|
| Calculator | 数学计算 | `Calculator: 2 + 3 * 4` |
| Weather | 天气查询 | `Weather: 北京` |
| CurrentTime | 当前时间 | `CurrentTime: (无参数)` |
| MedicalTerm | 医疗术语解释 | `MedicalTerm: 高血压` |
| DrugInfo | 药品信息查询 | `DrugInfo: 阿司匹林` |

## 工作流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  用户上传    │───▶│  文档解析    │───▶│  向量化存储  │
│   文档      │    │  (PDF/DOCX) │    │   (FAISS)   │
└─────────────┘    └─────────────┘    └─────────────┘
                                            │
                                            ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  返回回答    │◀───│  LLM 生成   │◀───│  文档检索    │
│   给用户    │    │  (GLM-4)   │    │  (相似度)   │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 技术架构

### 智能体架构

本项目采用 **Tool-Calling Agent（工具调用型智能体）** 架构：

1. **用户输入**：接收用户问题
2. **上下文构建**：结合对话历史和检索到的文档
3. **LLM 决策**：判断是否需要调用工具
4. **工具执行**：调用相应的工具获取结果
5. **回答生成**：综合所有信息生成最终回答

### RAG 流程

1. **文档加载**：解析 PDF/Word 文件
2. **文本分割**：将长文档分割成小片段
3. **向量化**：使用 TF-IDF 生成向量表示
4. **存储索引**：存入 FAISS 向量数据库
5. **语义检索**：根据用户问题检索相关片段
6. **增强生成**：将检索结果注入 Prompt，生成回答

## 扩展指南

### 添加新工具

在 `tools.py` 中添加新的工具函数：

```python
from langchain_core.tools import Tool
from pydantic import BaseModel, Field

class NewToolInput(BaseModel):
    param: str = Field(description="参数描述")

def new_tool(param: str) -> str:
    """工具描述"""
    # 工具逻辑
    return result

# 添加到 tools 列表
tools.append(
    Tool(
        name="NewTool",
        func=new_tool,
        description="工具用途描述",
        args_schema=NewToolInput
    )
)
```

### 更换模型

修改 `.env` 文件或 `app.py` 中的配置：

```env
MODEL=glm-4  # 或 gpt-3.5-turbo, gpt-4 等
BASE_URL=https://open.bigmodel.cn/api/paas/v4/
```

### 添加新的文档格式支持

在 `document_parser.py` 中添加新的解析方法：

```python
@staticmethod
def parse_txt(file_path):
    """解析 TXT 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
```

## 注意事项

- ⚠️ 需要有效的 API Key 才能正常运行
- ⚠️ 文档解析和向量化需要一定时间，请耐心等待
- ⚠️ 上传的文档内容需合法合规
- ⚠️ 系统回答仅供参考，不能替代专业医疗建议
- ⚠️ 建议使用 Python 3.8 或更高版本

## 常见问题

**Q: 启动报错 "No module named xxx"**
```bash
python -m pip install -r requirements.txt
```

**Q: API 调用失败**
- 检查 `.env` 文件中的 API Key 是否正确
- 确认网络连接正常
- 检查 BASE_URL 是否正确配置

**Q: 文档解析失败**
- 确认文档格式为 PDF 或 DOCX
- 检查文档是否加密或损坏
- 尝试将文档转换为纯文本格式

## 学习资源

- [LangChain 官方文档](https://python.langchain.com/docs/get_started/introduction)
- [FAISS 官方文档](https://github.com/facebookresearch/faiss)
- [Flask 官方文档](https://flask.palletsprojects.com/)
- [智谱 AI GLM 文档](https://open.bigmodel.cn/)

## License

MIT License
