# 医疗行业研报智能助手

这是一个专业的医疗行业研报智能助手项目，基于LangChain和大语言模型构建，能够自动解析医疗等行业研报/文档，并回答专业问题。

## 项目功能

- **文档解析**：支持PDF和Word格式的医疗研报解析
- **向量存储**：使用FAISS实现文档向量化和相似性搜索
- **专业问答**：基于文档内容和医疗知识回答专业问题
- **医疗工具**：集成医疗术语解释和药品信息查询工具
- **友好界面**：现代化的Web前端界面，支持文档上传和管理
- **多轮对话**：支持上下文理解和多轮对话

## 技术栈

- **后端**：Python 3.8+、Flask、LangChain
- **文档处理**：PyPDF2、python-docx
- **向量存储**：FAISS
- **前端**：HTML、CSS、JavaScript
- **大语言模型**：支持OpenAI API或其他兼容API

## 快速开始

### 1. 安装依赖

```bash
# 使用pip安装
python -m pip install -r requirements.txt

# 或者使用conda安装
conda create -n medical-agent python=3.8
conda activate medical-agent
conda install -c conda-forge langchain python-dotenv flask flask-cors pypdf2 python-docx faiss-cpu
```

### 2. 配置环境变量

复制`.env.example`文件并重命名为`.env`，然后填写你的API Key：

```env
# API Key
OPENAI_API_KEY=your-api-key
```

### 3. 运行项目

```bash
# 启动后端服务
python app.py

# 前端访问地址
# http://localhost:5000
```

### 4. 使用方法

1. **上传文档**：点击"选择医疗研报文件"按钮，上传PDF或Word格式的医疗研报
2. **查看文档**：上传成功后，文档会显示在文档列表中
3. **提问**：在输入框中输入关于医疗研报的问题，点击发送按钮或按回车键
4. **查看回答**：智能助手会基于文档内容和医疗知识提供专业回答
5. **管理文档**：可以点击文档列表中的"移除"按钮删除文档，或点击"清空文档"按钮移除所有文档
6. **清空历史**：点击"清空历史"按钮可以清除聊天记录

## 项目结构

```
agent-assistant/
├── frontend/           # 前端目录
│   └── index.html      # 前端界面
├── temp/               # 临时文件目录
├── app.py              # 后端API服务
├── main.py             # 命令行界面
├── document_parser.py  # 文档解析模块
├── vector_store.py     # 向量存储模块
├── tools.py            # 工具模块
├── requirements.txt    # 依赖配置
├── .env.example        # 环境变量示例
└── README.md           # 项目说明
```

## 核心模块说明

### MedicalReportAgent 类

- `__init__()`: 初始化LLM、文档解析器和向量存储
- `run(user_input)`: 处理用户输入，检索相关文档并生成响应
- `add_document(file_path)`: 添加文档到系统并进行向量化
- `clear_documents()`: 清空所有文档
- `clear_history()`: 清空对话历史

### 文档处理流程

1. 上传文档到系统
2. 解析文档内容（PDF或Word）
3. 分割文档为小片段
4. 使用嵌入模型生成向量
5. 存储向量到FAISS索引
6. 用户提问时，检索相关文档片段
7. 结合文档内容和医疗知识生成回答

## 扩展建议

1. **添加更多医疗专业工具**：如疾病查询、治疗方案推荐等
2. **集成专业医疗知识库**：提高回答的准确性和专业性
3. **优化向量存储**：支持更大规模的文档管理
4. **添加用户认证**：实现个性化服务和多用户支持
5. **部署到云端**：提供在线服务

## 注意事项

- 需要有效的API Key
- 文档解析和向量化可能需要一定时间，取决于文档大小
- 请确保上传的文档内容合法合规
- 系统生成的回答仅供参考，不能替代专业医疗建议

## 学习资源

- [LangChain 官方文档](https://python.langchain.com/docs/get_started/introduction)
- [FAISS 官方文档](https://github.com/facebookresearch/faiss)
- [Flask 官方文档](https://flask.palletsprojects.com/)
- [Python 官方文档](https://docs.python.org/3/)
