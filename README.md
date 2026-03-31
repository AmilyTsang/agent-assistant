# 智能助手 Agent 项目

这是一个入门级的智能助手Agent项目，基于LangChain和OpenAI API构建。

## 项目功能

- 基于GPT-3.5-turbo的对话能力
- 支持多轮对话
- 友好的命令行界面
- 支持清空对话历史

## 技术栈

- Python 3.8+
- LangChain
- OpenAI API
- Rich (终端美化)

## 快速开始

### 1. 安装依赖

```bash
# 使用pip安装
python -m pip install langchain langchain-openai python-dotenv rich

# 或者使用conda安装
conda install -c conda-forge langchain python-dotenv rich openai
```

### 2. 配置环境变量

复制`.env.example`文件并重命名为`.env`，然后填写你的OpenAI API Key：

```env
# OpenAI API Key
OPENAI_API_KEY=your-openai-api-key
```

### 3. 运行项目

```bash
python main.py
```

### 4. 使用方法

- 输入问题与助手对话
- 输入`exit`退出程序
- 输入`clear`清空对话历史

## 项目结构

```
agent/
├── main.py          # 主程序文件
├── requirements.txt # 依赖配置
├── .env.example     # 环境变量示例
└── README.md        # 项目说明
```

## 核心代码说明

### SimpleAgent 类

- `__init__()`: 初始化LLM、提示模板和对话历史
- `run(user_input)`: 处理用户输入并返回响应
- `clear_history()`: 清空对话历史

### 对话流程

1. 加载环境变量和初始化LLM
2. 接收用户输入
3. 构建对话历史上下文
4. 调用LLM生成响应
5. 更新对话历史
6. 显示响应给用户

## 扩展建议

1. **添加工具调用能力**：集成外部工具如搜索引擎、计算器等
2. **实现记忆机制**：使用向量数据库存储对话历史
3. **添加多语言支持**：根据用户语言自动切换回复语言
4. **实现特定领域知识**：针对特定领域（如编程、数学等）进行优化

## 注意事项

- 需要有效的OpenAI API Key
- 对话历史会保存在内存中，程序重启后会丢失
- 请遵守OpenAI的使用条款和限制

## 学习资源

- [LangChain 官方文档](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI API 文档](https://platform.openai.com/docs/introduction)
- [Python 官方文档](https://docs.python.org/3/)
