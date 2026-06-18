# 🧠 多模态 AI 智能体
Multimodal AI Agent with Vision & RAG · 本地部署 · 全链路智能

**一个集成了视觉理解、知识库检索、工具调用与多轮对话能力的本地 AI 智能体。**

## 硬件要求

- **推荐配置**：≥ 8GB 显存的 NVIDIA GPU（RTX 3060 / 4060 / 5060 及以上）
- **测试环境**：RTX 5060 Ti (16GB) + 32GB 内存
- **模型规格**：Qwen3-VL-8B-Instruct (Q4_K_M 量化)
- **显存占用**：约 4.8-5.5 GB（含上下文缓存）
- **最低运行**：6GB 显存（需限制上下文长度）
- **4GB 显存**：无法运行 8B 模型，建议改用 Qwen3-VL-2B 或 API 中转方案

## ✨ 核心能力

| 能力 | 说明 |
|:---|:---|
| 👁️ 视觉理解 | 基于 Qwen3-VL 多模态模型，能"看懂"图片并生成 300+ 字详细描述 |
| 📚 知识库问答 (RAG) | 基于 Chroma 向量数据库 + 本地文档，实现检索增强生成 |
| 🛠️ 工具调用 (ReAct) | 支持时间查询、数学计算、图片描述等自定义工具 |
| 💬 多轮对话记忆 | 基于 LangChain 的消息记忆机制，支持上下文理解与身份记忆 |
 

## 📦 技术栈

| 组件 | 用途 |
|:---|:---|
| **LangChain (Agent 框架)** | 工具绑定、ReAct 循环与多轮对话管理 |
| **FastAPI + Uvicorn** | HTTP 服务封装与异步接口 |
| **Ollama** | 本地大模型服务（支持 CPU/GPU） |
| **Qwen3-VL (8B Q4)** | 多模态视觉语言模型（图片理解） |
| **Chroma** | 轻量级向量数据库（RAG 检索） |
| **nomic-embed-text** | 文本向量化嵌入模型 |
| **Docker + Docker Compose** | 容器化部署与环境管理 | 
| **Python 3.10+** | 开发语言 |

## 📁 项目结构

    agent_project/
    ├── app.py # FastAPI 服务版（主程序）
    ├── config.py # 配置文件（支持环境变量）
    ├── first_agent.py # 终端版 Agent（原始版本）
    ├── agent_with_memory.py # 短期记忆程序
    ├── build_vector_store.py # 构建知识库程序
    ├── requirements.txt # 依赖清单
    ├── Dockerfile # Docker 镜像构建文件
    ├── docker-compose.yaml # Docker Compose 一键部署
    ├── .env.example # 环境变量配置模板
    ├── chroma_db/ # Chroma 向量数据库持久化目录
    ├── screenshots/ # 运行效果截图/测试图
    │ ├── terminal_demo.png
    │ ├── terminal_demo_v2.png
    │ ├── terminal_demo_v3.png
    │ └── terminal_demo_v4.png
    └── README.md # 文档
        

## 🚀 快速开始

### 方式一：本地运行

#### 1. 环境准备

    # 克隆项目（或直接下载代码）
    git clone <your-repo-url>
    cd agent_project

    # 创建虚拟环境（推荐）
    python -m venv venv
    source venv/bin/activate      # Linux/Mac
    # 或
    venv\Scripts\activate         # Windows

    # 安装依赖
    pip install -r requirements.txt

#### 2. 启动 Ollama 服务

    # 启动 Ollama 服务（确保已安装）
    ollama serve

    # 拉取所需模型（如果尚未下载）
    ollama pull qwen3-vl:8b-instruct-q4_K_M   # 多模态视觉模型
    ollama pull nomic-embed-text               # 向量化嵌入模型

#### 3. 准备知识库（可选）

将你的文本文档（`.txt` / `.md`）放入 放在任意目录下，然后运行 `build_vector_store.py` 构建向量索引：

```bash
python build_vector_store.py
```
该脚本会读取指定目录下的文本文件，生成向量并存入 chroma_db/ 目录。如果你不需要 RAG 功能，可以跳过此步。

#### 4. 运行 Agent

    python first_agent.py

在终端中输入问题，Agent 会自动调用相应工具或检索知识库。

#### 5. 启动 HTTP 服务

```bash
    python app.py
```
服务默认运行在 http://localhost:8000

#### 6. 调用API
Post /chat —— 发送对话请求
```bash
    curl -X POST "http://localhost:8000/chat" \
         -H "Content-Type: application/json" \
         -d '{"question":"现在几点了","session_id":"demo"}'
```
响应示例：
{"code":200,"data":"现在是2026年6月17日上午10点54分。","msg":"success"}

GET /health —— 健康检查
```bash
    curl http://localhost:8000/health
    # {"status":"ok"}
```

### 方式二：Docker 一键部署（推荐）

#### 1. 克隆项目

```bash
    git clone <your-repo-url>
    cd agent_project
```

#### 2. 配置环境变量

复制 .env.example 为 .env，并修改 OLLAMA_BASE_URL 为你的宿主机 IP
```bash
    cp .env.example .env
```
```properties
    # .env
    OLLAMA_MODEL=qwen3-vl:8b-instruct-q4_K_M
    EMBEDDING_MODEL=nomic-embed-text
    OLLAMA_BASE_URL=http://192.168.x.x:11434   # 替换为你的宿主机 IP
    PORT=8000
    HOST=0.0.0.0
```

#### 3. 启动服务

```bash
    docker-compose up -d
```
查看日志：
```bash
    docker-compose logs
```
停止服务:
```bash
    docker-compose down
```

#### 4. 测试
```bash
    curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d "{\"question\":\"现在几点了\",\"session_id\":\"test\"}" 
```

#### 5. 调用API
POST /chat —— 发送对话请求
```bash
    curl -X POST "http://localhost:8000/chat" \
         -H "Content-Type: application/json" \
         -d '{"question":"现在几点了","session_id":"demo"}'
```
响应示例：
{"code":200,"data":"现在是2026年6月17日上午10点54分。","msg":"success"}
GET /health —— 健康检查
```bash
    curl http://localhost:8000/health
    # {"status":"ok"}
```

## 📸 运行效果

::: screenshot-placeholder
**🖼️ API 测试截图**\
（请将 `terminal_demo_v4.png` 放入 `screenshots/`
目录，此处将自动显示）\
[实际文档中请替换为真实图片]{style="font-size:0.85rem; color:#94a3b8;"}
![API 测试截图](./screenshots/api_demo.png)
:::

**测试用例一览：**

| 用户输入 | Agent 输出 | 验证能力 |
|:---------|:-----------|:---------|
| `现在几点了` | *现在是2026年6月17日上午10点54分* | ✅ 工具调用（时间） |
| `45+62=？` | *107* | ✅ 工具调用（计算） |
| `描述一下 screenshots/terminal_demo.png` | *300+ 字详细描述（人物、服饰、背景、风格）* | ✅ 多模态视觉理解 |
| `我是欧阳超` | *擅长视觉和边缘计算* | ✅ RAG 检索 |
| `我的学习路径` | *你的学习路线分为三个阶段（知识库内容）* | ✅ RAG 检索增强 |
| `我是谁` | *你是欧阳超，一名 AI 工程师…* | ✅ 多轮对话记忆 |

## ⚠️ 测试注意事项

### 关于图片描述功能的测试

在 Windows 系统下，直接使用 `curl` 命令行发送包含中文和路径的 JSON 时，可能因转义问题导致图片路径解析失败，返回“文件不存在”。

**推荐方式**：将请求体保存为 `payload.json` 文件，再通过 `curl` 发送：

```bash
# 1. 创建 payload.json
echo {"question":"描述一下 screenshots/terminal_demo.png","session_id":"test"} > payload.json

# 2. 发送请求
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d @payload.json
```

## ❓ 常见问题
### 1. Docker 容器启动后反复重启
检查容器日志：
```bash
    docker logs agent-api --tail 50
```
常见原因：app.py 语法错误或缺少依赖。修复后重新构建：
```bash
    docker-compose up --build -d
```

### 2. 容器内访问不了宿主机的 Ollama
确保 Ollama 监听 0.0.0.0：
```bash
    set OLLAMA_HOST=0.0.0.0
    ollama serve
```
并在 .env 中正确设置 OLLAMA_BASE_URL 为宿主机 IP。

### 3. 图片描述返回“文件不存在”
·确认 screenshots/ 目录已正确挂载（Docker Compose 已默认挂载）
·确认图片文件确实存在于该目录
·使用 payload.json 方式发送请求（见上方说明）

### 4. 如何切换模型？
修改 .env 中的 OLLAMA_MODEL 变量，然后重启服务：
```bash
    docker-compose down
    docker-compose up -d
```
确保新模型已在 Ollama 中下载：
```bash
    ollama pull <新模型名>
```

### 5. 端口被占用
修改 .env 中的 PORT 变量，或直接修改 docker-compose.yaml 中的端口映射：
```yaml
    ports:
    - "8001:8000"   # 将宿主机端口改为 8001
```

## 📌 后续规划（迭代方向）

- ✅ 基础 Agent 框架（工具调用 + RAG + 多模态视觉）
- ✅ 时间格式优化（24h → 12h 制）
- ✅ 图片路径支持相对路径 + 异常捕获
- ✅ FastAPI 封装为 HTTP 服务
- ✅ 多会话记忆（session_id 隔离）
- ✅ 配置文件分离（config.py）
- ✅ Docker 容器化部署（docker-compose）
- ⏳ 生产级会话存储（Redis 替代内存字典）
- ⏳ 异步接口支持（解决大图推理阻塞）
- ⏳ 结构化日志（Loguru）替代 print 调试
- ⏳ CI/CD 流水线（GitHub Actions）

## 📄 License
    MIT © 2026 阮晨希

## 🤝 贡献与反馈
    欢迎通过 Issue 或 Pull Request 提出改进建议。

------------------------------------------------------------------------


