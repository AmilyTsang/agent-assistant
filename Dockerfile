# ==================================================
# 医疗智能助手 Dockerfile
# ==================================================
# 使用 Python 3.10（支持 PEFT 0.19.1）
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖（使用国内镜像加速）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 安装 PEFT 0.19.1（需要 Python 3.10+）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple peft>=0.19.1

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p temp data

# 设置环境变量
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PORT=5000

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "app.py"]
