# 医疗行业研报智能助手 (Medical Report Agent)

基于 LangChain 和大语言模型构建的智能医疗研报分析助手，支持文档解析、专业问答、LoRA 微调、工具调用等功能。

---

## ✨ 项目功能

| 功能模块 | 描述 |
|---------|------|
| **文档解析** | 支持 PDF 和 Word 格式的医疗研报解析 |
| **向量检索** | 基于 FAISS 的高效向量相似度检索 |
| **专业问答** | 基于 RAG（检索增强生成）架构，结合文档内容和专业知识回答问题 |
| **工具集** | 内置计算器、天气查询、时间查询、医疗术语解释、药品信息查询等工具 |
| **智能体架构** | 基于 LangChain Agent 的工具调用型智能体，支持自主规划和多轮对话 |
| **LoRA 微调** | 支持使用医疗术语数据集进行领域微调，提升医疗术语解释能力 |
| **双模式运行** | 支持 API 调用（GLM-5）和本地 LoRA 模型（GLM-4-6B）两种模式 |
| **Web 界面** | 现代化的前端界面，支持文档上传、对话管理 |
| **命令行模式** | 支持命令行交互模式，方便调试和快速测试 |

---

## 🛠️ 技术栈

### 后端技术
| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.8+ | 开发语言 |
| Flask | 2.0+ | Web 框架 |
| Flask-CORS | - | 跨域支持 |
| LangChain | 0.3+ | 智能体框架 |
| LangChain-OpenAI | - | OpenAI API 集成 |
| FAISS | - | 向量数据库 |
| PyPDF2 | - | PDF 解析 |
| python-docx | - | Word 解析 |
| scikit-learn | - | TF-IDF 嵌入 |
| pydantic | <2.0 | 数据验证 |
| rich | - | 命令行美化 |
| python-dotenv | - | 环境变量管理 |

### AI/ML 技术
| 技术 | 用途 |
|------|------|
| GLM-5 / GLM-4-6B | 大语言模型 |
| LoRA (PEFT) | 参数高效微调 |
| TF-IDF | 文本向量化 |
| FAISS | 向量相似度检索 |
| LangChain Agent | 工具调用型智能体 |

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd agent-assistant

# 安装基础依赖
python -m pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制示例配置文件
copy .env.example .env  # Windows
# 或
cp .env.example .env    # Linux/Mac
```

编辑 `.env` 文件：

```env
# API 模式配置（默认）
OPENAI_API_KEY=your-api-key-here
MODEL=glm-5v-turbo
BASE_URL=https://open.bigmodel.cn/api/paas/v4/
TEMPERATURE=0.7

# 本地 LoRA 模式配置（可选）
USE_LOCAL_MODEL=false
LORA_PATH=./lora_medical_terms

# Agent 配置
ENABLE_PLANNING=true
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
- **健康检查**：`curl http://localhost:5000/health`

---

## 📊 LoRA 微调（可选）

### 1. 安装微调依赖

```bash
pip install torch transformers accelerate peft datasets evaluate sentencepiece
```

### 2. 执行微调

```bash
# 使用医疗术语数据集训练
python train_lora_medical_terms.py

# 或使用通用数据集训练
python train_lora.py
```

### 3. 配置本地模型

在 `.env` 中启用本地模式：

```env
USE_LOCAL_MODEL=true
LORA_PATH=./lora_medical_terms
```

### 4. 测试微调效果

```bash
python medical_terms_inference.py
```

---

## 🐳 Docker 部署

### 快速启动

```bash
# 1. 复制环境变量配置
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac

# 2. 编辑 .env 文件，填入 API Key

# 3. 启动服务（CPU 版本）
docker-compose up -d

# 或启动 GPU 版本（需安装 NVIDIA Docker）
docker-compose -f docker-compose.gpu.yml up -d

# 4. 验证服务
curl http://localhost:5000/health
```

### Docker 配置文件

| 文件 | 说明 |
|------|------|
| `Dockerfile` | CPU 版本镜像配置 |
| `Dockerfile.gpu` | GPU 版本镜像配置（支持 LoRA 训练） |
| `docker-compose.yml` | CPU 版本编排文件 |
| `docker-compose.gpu.yml` | GPU 版本编排文件 |

### 健康检查

```bash
# 检查服务状态
curl http://localhost:5000/health
# 返回示例：{"status": "healthy", "model": "glm-5v-turbo", ...}
```

---

## 📁 项目结构

```
agent-assistant/
├── frontend/                     # 前端文件
│   └── index.html               # Web 界面主页面
├── data/                        # 数据集目录
│   └── medical_finetune.json    # 医疗术语微调数据集
├── __pycache__/                 # Python 缓存文件
├── .idea/                       # IDE 配置
├── app.py                       # Flask Web 服务入口
├── main.py                      # 命令行模式入口
├── document_parser.py           # 文档解析模块
├── vector_store.py              # 向量存储与检索模块
├── tools.py                     # 工具定义模块（集成 LoRA）
├── train_lora.py                # 通用 LoRA 训练脚本
├── train_lora_medical_terms.py  # 医疗术语专项训练脚本
├── lora_inference.py            # 通用推理脚本
├── medical_terms_inference.py   # 医疗术语推理脚本
├── test_simple.py               # 基础功能测试脚本
├── test_agent.py                # 完整功能测试脚本
├── requirements.txt             # Python 依赖
├── Dockerfile                   # Docker CPU 镜像配置
├── Dockerfile.gpu               # Docker GPU 镜像配置
├── docker-compose.yml           # Docker Compose 配置
├── docker-compose.gpu.yml       # Docker Compose GPU 配置
├── .env                         # 环境变量配置
├── .env.example                 # 环境变量示例
├── README.md                    # 项目说明文档
└── 部署.md                      # 完整部署文档
```

---

## 🧠 Agent 能力介绍

### 自主规划能力
- 支持多步骤任务规划
- 基于 `AgentExecutor` 的规划执行
- 反思机制和错误重试（最多 3 次）
- 双层记忆结构（短期记忆 + 历史摘要）

### 工具调用能力
| 工具名称 | 功能 | 示例 |
|---------|------|------|
| Calculator | 数学计算 | `Calculator: 2 + 2 * 3` |
| Weather | 天气查询 | `Weather: 北京` |
| CurrentTime | 当前时间查询 | `CurrentTime: (无参数)` |
| MedicalTerm | 医疗术语解释（支持 LoRA 模型） | `MedicalTerm: 高血压` |
| DrugInfo | 药品信息查询 | `DrugInfo: 阿司匹林` |

### 记忆管理
- 对话历史长度限制（默认 20 轮）
- 历史摘要压缩机制（超过 5 轮自动压缩）
- 支持手动清空对话历史

---

## 🔧 使用方法

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

---

## 🏗️ 核心模块说明

### MedicalReportAgent 类

智能体核心类，负责处理用户输入和生成回答。

| 方法 | 功能 |
|------|------|
| `__init__(api_key, model, temperature, base_url, use_local_model, lora_path)` | 初始化 LLM、文档解析器、向量存储 |
| `run(user_input)` | 处理用户输入，检索文档并生成响应 |
| `_run_with_planning(user_input)` | 带规划能力的执行流程（API 模式） |
| `_run_basic(user_input)` | 基础执行模式（本地模型） |
| `add_document(file_path)` | 添加文档到系统并进行向量化 |
| `clear_documents()` | 清空所有已添加的文档 |
| `clear_history()` | 清空对话历史记录 |

### 工具模块 (tools.py)

集成了基于 LoRA 的医疗术语解释器：

```python
class MedicalTermExplainer:
    """医疗术语解释器（基于 LoRA 微调模型）"""
    def __init__(self, base_model_name="THUDM/glm-4-6b-chat", lora_path="./lora_medical_terms"):
        # 加载基础模型和 LoRA 权重
        ...
    
    def explain(self, term):
        # 使用 LoRA 模型解释医疗术语
        ...
```

---

## 🔄 工作流程

### RAG + Agent 流程

```
用户输入 → 任务分析 → 文档检索 → LLM 决策 → 工具调用 → 结果汇总 → 返回回答
```

### LoRA 微调流程

```
数据集准备 → 模型加载 → LoRA 配置 → 训练执行 → 权重保存 → 推理部署
```

---

## 📈 扩展指南

### 添加新工具

在 `tools.py` 中添加新的工具函数：

```python
class NewToolInput(BaseModel):
    param: str = Field(description="参数描述")

def new_tool(param: str) -> str:
    """工具描述"""
    return result

tools.append(Tool(
    name="NewTool",
    func=new_tool,
    description="工具用途描述",
    args_schema=NewToolInput
))
```

### 配置本地 LoRA 模型

1. 先执行训练：`python train_lora_medical_terms.py`
2. 在 `.env` 中配置：

```env
USE_LOCAL_MODEL=true
LORA_PATH=./lora_medical_terms
```

---

## ⚠️ 注意事项

- 需要有效的 API Key 才能使用 API 模式
- LoRA 训练需要 GPU（建议显存 ≥8GB）
- 文档解析和向量化需要一定时间，请耐心等待
- 上传的文档内容需合法合规
- 系统回答仅供参考，不能替代专业医疗建议
- 建议使用 Python 3.8 或更高版本
- 天气查询功能需要配置 OpenWeatherMap API Key

---

## ❓ 常见问题

**Q: 启动报错 "No module named xxx"**
```bash
python -m pip install -r requirements.txt
```

**Q: API 调用失败**
- 检查 `.env` 文件中的 API Key 是否正确
- 确认网络连接正常
- 检查 BASE_URL 是否正确配置

**Q: LoRA 训练显存不足**
- 使用较小的模型（如 GLM-4-6B）
- 减小 batch_size（默认 2）
- 启用梯度累积

**Q: 文档解析失败**
- 确认文档格式为 PDF 或 DOCX
- 检查文档是否加密或损坏

**Q: 天气查询功能不工作**
- 需要在 `tools.py` 中配置 OpenWeatherMap API Key

---

## 📚 学习资源

- [LangChain 官方文档](https://python.langchain.com/docs/get_started/introduction)
- [PEFT (LoRA) 文档](https://huggingface.co/docs/peft)
- [FAISS 官方文档](https://github.com/facebookresearch/faiss)
- [Flask 官方文档](https://flask.palletsprojects.com/)
- [智谱 AI GLM 文档](https://open.bigmodel.cn/)

---

## 📄 License

MIT License

---

**项目代码**：https://github.com/AmilyTsang/agent-assistant