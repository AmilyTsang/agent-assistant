# AI Agent 从原生到框架学习路径

## 📚 学习路线总览

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI Agent 学习路线                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  阶段1          阶段2          阶段3          阶段4              │
│  ┌─────┐       ┌─────┐       ┌─────┐       ┌─────┐            │
│  │基础 │  ──▶  │工具 │  ──▶  │ReAct│  ──▶  │框架 │            │
│  │Agent│       │调用 │       │模式 │       │实战 │            │
│  └─────┘       └─────┘       └─────┘       └─────┘            │
│                                                                 │
│  理解LLM调用    工具注册与     推理+行动      LangChain          │
│  基础对话       参数解析       循环执行       生产级开发          │
│                                                                 │
│  难度: ★☆☆☆☆   难度: ★★☆☆☆   难度: ★★★☆☆   难度: ★★★☆☆        │
│  时间: 30分钟   时间: 45分钟   时间: 60分钟   时间: 45分钟        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 阶段 1：原生实现基础 Agent

### 学习目标
- 理解 LLM API 调用的本质
- 掌握消息结构（system/user/assistant）
- 实现最简单的对话 Agent

### 项目结构
```
agent-tutorial/
├── stage1-basic/
│   ├── 01_hello_llm.py        # 第一次调用 LLM
│   ├── 02_conversation.py     # 多轮对话
│   └── 03_simple_agent.py     # 简单 Agent 封装
```

### 复制以下提示词到 Claude

---

**【提示词 1-1】基础 LLM 调用**

```
请帮我创建一个 Python 文件 `01_hello_llm.py`，实现：

1. 使用 OpenAI API 调用 GPT-4（支持通过环境变量 OPENAI_API_KEY 配置）
2. 发送一条简单的用户消息 "你好，请介绍一下你自己"
3. 打印模型的回复
4. 包含完整的错误处理

要求：
- 使用 requests 库直接调用 API（不用 SDK），帮我理解 HTTP 请求的本质
- 代码要有详细的中文注释
- 打印请求的完整过程（发送什么、收到什么）
```

---

**【提示词 1-2】多轮对话**

```
请帮我创建一个 Python 文件 `02_conversation.py`，实现：

1. 一个可以持续对话的聊天程序
2. 维护消息历史（messages 列表）
3. 用户输入 "quit" 时退出
4. 每次对话都带上历史消息

要求：
- 展示 messages 列表如何累积
- 打印当前对话轮数
- 用不同颜色区分用户输入和 AI 回复（可以用 colorama 库）
- 解释为什么需要维护消息历史
```

---

**【提示词 1-3】简单 Agent 封装**

```
请帮我创建一个 Python 文件 `03_simple_agent.py`，实现：

1. 一个 SimpleAgent 类，封装对话逻辑
2. 包含以下属性：
   - name: Agent 名称
   - system_prompt: 系统提示词
   - messages: 消息历史
3. 包含以下方法：
   - chat(user_input): 发送消息并获取回复
   - reset(): 重置对话
   - show_history(): 显示对话历史

使用示例：
```python
agent = SimpleAgent(
    name="小助手",
    system_prompt="你是一个友好的AI助手，回答简洁有趣。"
)
agent.chat("你好！")
agent.chat("我刚才说了什么？")  # 测试记忆
agent.show_history()
```

要求：
- 代码结构清晰，面向对象
- 包含类型注解
- 添加使用示例和测试代码
```

---

## 🛠️ 阶段 2：工具调用 Agent

### 学习目标
- 理解 Function Calling 的工作原理
- 学会定义工具（函数）规范
- 实现 Agent 自动选择和调用工具

### 项目结构
```
agent-tutorial/
├── stage2-tools/
│   ├── 01_function_schema.py   # 工具定义规范
│   ├── 02_tool_execution.py    # 工具执行机制
│   └── 03_tool_agent.py        # 完整工具 Agent
```

### 复制以下提示词到 Claude

---

**【提示词 2-1】理解 Function Calling**

```
请帮我创建一个 Python 文件 `01_function_schema.py`，实现：

1. 定义两个工具函数：
   - get_weather(city: str) -> str: 查询天气（模拟返回）
   - calculate(expression: str) -> float: 计算数学表达式

2. 将函数转换为 OpenAI Function Calling 需要的 JSON Schema 格式
3. 发送带有 tools 参数的请求
4. 解析模型返回的 function_call
5. 执行函数并返回结果

要求：
- 详细展示 tools 参数的 JSON 结构
- 打印每一步的数据流（请求 → 响应 → 解析 → 执行）
- 解释 function_call 和 function_call_output 的关系
- 用流程图注释展示整个调用链路
```

---

**【提示词 2-2】工具执行机制**

```
请帮我创建一个 Python 文件 `02_tool_execution.py`，实现：

1. 一个 ToolExecutor 类，负责：
   - 注册工具函数
   - 验证参数
   - 执行函数
   - 处理错误

2. 支持的工具：
   - search_web(query): 网页搜索（模拟）
   - get_current_time(): 获取当前时间
   - calculate(expression): 数学计算

3. 实现自动参数转换（字符串 → 正确类型）

使用示例：
```python
executor = ToolExecutor()
executor.register(search_web)
executor.register(get_current_time)

# 执行工具调用
result = executor.execute("search_web", {"query": "Python教程"})
```

要求：
- 使用 Python 装饰器 @tool 来注册函数
- 自动从函数签名生成 JSON Schema
- 包含完整的错误处理
- 添加执行日志
```

---

**【提示词 2-3】完整工具 Agent**

```
请帮我创建一个 Python 文件 `03_tool_agent.py`，实现：

1. 一个 ToolAgent 类，继承自阶段1的 SimpleAgent
2. Agent 能自动判断何时需要调用工具
3. 支持多轮工具调用（一个任务可能需要多个工具）
4. 最终整合工具结果回答用户

工作流程：
用户输入 → LLM 判断 → 选择工具 → 执行工具 → 返回结果给 LLM → 生成回答

使用示例：
```python
agent = ToolAgent(
    name="智能助手",
    tools=[search_web, get_weather, calculate]
)

# 单工具调用
agent.run("今天北京天气怎么样？")

# 多工具调用
agent.run("帮我计算 123 * 456，然后搜索这个数字的含义")

# 无需工具
agent.run("给我讲个笑话")
```

要求：
- 实现完整的工具调用循环
- 支持并行工具调用（如果 LLM 返回多个 tool_calls）
- 打印详细的决策过程（为什么选择这个工具）
- 最大迭代次数限制，防止无限循环
```

---

## 🔄 阶段 3：ReAct 模式实现

### 学习目标
- 深入理解 ReAct (Reasoning + Acting) 模式
- 实现 Thought-Action-Observation 循环
- 构建能处理复杂任务的 Agent

### 项目结构
```
agent-tutorial/
├── stage3-react/
│   ├── 01_thought_action.py    # 思考与行动
│   ├── 02_observation.py       # 观察与反思
│   └── 03_react_agent.py       # 完整 ReAct Agent
```

### 复制以下提示词到 Claude

---

**【提示词 3-1】Thought-Action 模式**

```
请帮我创建一个 Python 文件 `01_thought_action.py`，实现：

1. 一个 ReAct 提示词模板，引导模型输出：
   Thought: [思考过程]
   Action: [工具名称]
   Action Input: [工具参数]

2. 解析模型输出，提取 Thought、Action、Action Input

3. 支持的动作：
   - Search[query]: 搜索信息
   - Calculate[expression]: 计算表达式
   - Finish[answer]: 完成任务，给出最终答案

示例对话：
用户: "北京和上海今天天气哪个更好？"
Agent:
  Thought: 我需要分别查询北京和上海的天气，然后比较
  Action: Search
  Action Input: 北京今天天气
...

要求：
- 使用少样本示例(Few-shot)引导模型输出格式
- 实现可靠的输出解析（正则表达式）
- 打印每一步的 Thought 和 Action
```

---

**【提示词 3-2】Observation 循环**

```
请帮我创建一个 Python 文件 `02_observation.py`，实现：

1. 完成 ReAct 循环的 Observation 部分：
   - 执行 Action
   - 获取结果
   - 将结果注入下一轮对话

2. 循环流程：
```
┌─────────────────────────────────────────┐
│                                         │
│  Thought → Action → Observation         │
│     ↑                    │              │
│     └────────────────────┘              │
│              ↓                          │
│         Finish（最终答案）               │
│                                         │
└─────────────────────────────────────────┘
```

3. 实现终止条件检测：
   - 检测到 Finish 动作
   - 达到最大迭代次数
   - 模型说"我无法完成"

要求：
- 打印完整的思考链路
- 支持中途暂停和继续
- 记录所有 Observation 用于调试
```

---

**【提示词 3-3】完整 ReAct Agent**

```
请帮我创建一个 Python 文件 `03_react_agent.py`，实现：

1. 一个完整的 ReActAgent 类

2. 核心方法：
   - think(): 生成 Thought
   - act(): 执行 Action
   - observe(): 获取 Observation
   - run(): 完整执行循环

3. 支持的工具：
   - search(query): 搜索信息
   - wikipedia(topic): 查询维基百科（模拟）
   - calculator(expr): 数学计算
   - finish(answer): 完成任务

4. 复杂任务测试用例：
```python
agent = ReActAgent(tools=[search, wikipedia, calculator])

# 多步骤推理任务
agent.run("爱因斯坦获得诺贝尔奖时几岁？")
# 需要：搜索爱因斯坦出生年份 → 搜索获奖年份 → 计算

agent.run("比较 Python 和 JavaScript 的创建年份差")
# 需要：搜索 Python 年份 → 搜索 JS 年份 → 计算差值
```

要求：
- 完整的执行日志，展示推理过程
- 支持从中间状态恢复
- 最大步数限制和超时处理
- 导出完整的思考链路为 JSON
```

---

## 🏗️ 阶段 4：LangChain 框架实战

### 学习目标
- 掌握 LangChain 核心概念
- 使用 LangGraph 构建复杂工作流
- 对比原生实现与框架的差异

### 项目结构
```
agent-tutorial/
├── stage4-langchain/
│   ├── 01_langchain_basics.py    # LangChain 基础
│   ├── 02_langchain_tools.py     # 工具集成
│   └── 03_langgraph_agent.py     # LangGraph 实现
```

### 复制以下提示词到 Claude

---

**【提示词 4-1】LangChain 基础**

```
请帮我创建一个 Python 文件 `01_langchain_basics.py`，实现：

1. LangChain 核心概念演示：
   - ChatOpenAI: LLM 封装
   - PromptTemplate: 提示词模板
   - LLMChain: 简单链
   - ConversationChain: 对话链

2. 对比原生实现：
```python
# 原生方式（阶段1）
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "你好"}]
)

# LangChain 方式
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
chain = prompt | llm
chain.invoke({"input": "你好"})
```

3. 实现带记忆的对话

要求：
- 解释每个组件的作用
- 展示 LangChain 如何简化代码
- 对比两种实现的代码量
```

---

**【提示词 4-2】LangChain 工具集成**

```
请帮我创建一个 Python 文件 `02_langchain_tools.py`，实现：

1. 使用 @tool 装饰器定义工具：
```python
from langchain.tools import tool

@tool
def search_web(query: str) -> str:
    """搜索网页信息"""
    return f"搜索结果: {query}"

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}今天是晴天"
```

2. 创建 Agent 并绑定工具：
```python
from langchain.agents import create_tool_calling_agent, AgentExecutor

tools = [search_web, get_weather]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
```

3. 对比阶段2的原生实现

要求：
- 展示 LangChain 如何自动处理工具调用循环
- 实现自定义工具
- 添加工具调用日志
```

---

**【提示词 4-3】LangGraph 实现**

```
请帮我创建一个 Python 文件 `03_langgraph_agent.py`，实现：

1. 使用 LangGraph 构建 ReAct Agent：
```python
from langgraph.graph import StateGraph, END

# 定义状态
class AgentState(TypedDict):
    messages: List[BaseMessage]
    thoughts: List[str]
    actions: List[str]

# 定义节点
def think(state): ...
def act(state): ...
def observe(state): ...

# 构建图
graph = StateGraph(AgentState)
graph.add_node("think", think)
graph.add_node("act", act)
graph.add_node("observe", observe)

# 定义边
graph.add_edge("think", "act")
graph.add_edge("act", "observe")
graph.add_conditional_edges("observe", should_continue, {
    "continue": "think",
    "end": END
})
```

2. 可视化执行流程：
```python
from IPython.display import Image
Image(graph.get_graph().draw_mermaid_png())
```

3. 对比阶段3的原生 ReAct 实现

要求：
- 解释 StateGraph 的概念
- 展示如何定义条件边
- 实现状态检查点（可以从任意状态恢复）
- 对比代码复杂度
```

---

## 📋 学习检查清单

完成每个阶段后，确认你已掌握：

### 阶段 1 ✅
- [ ] 能独立调用 LLM API
- [ ] 理解 messages 数组的作用
- [ ] 能封装简单的 Agent 类

### 阶段 2 ✅
- [ ] 理解 Function Calling 的 JSON Schema
- [ ] 能定义和注册工具函数
- [ ] 理解工具调用的完整流程

### 阶段 3 ✅
- [ ] 理解 ReAct 的核心思想
- [ ] 能实现 Thought-Action-Observation 循环
- [ ] 能处理多步骤复杂任务

### 阶段 4 ✅
- [ ] 掌握 LangChain 核心组件
- [ ] 能使用 LangGraph 构建工作流
- [ ] 理解框架带来的价值和代价

---

## 🎓 进阶方向

完成以上4个阶段后，你可以继续探索：

1. **CrewAI 多 Agent 协作** - 让多个 Agent 像团队一样工作
2. **RAG 增强** - 让 Agent 能查询知识库
3. **记忆系统** - 实现长期记忆和个性化
4. **MCP 协议** - 接入 Claude 的工具标准
5. **部署上线** - 将 Agent 部署为 API 服务

---

## 💡 学习建议

1. **每个阶段都动手实践**，不要只是看代码
2. **尝试修改代码**，看看会发生什么
3. **添加 print 语句**，观察数据流动
4. **遇到错误不要慌**，这是学习的机会
5. **对比原生和框架实现**，理解"为什么要有框架"

---

祝学习愉快！🚀