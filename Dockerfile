# 使用 Python 3.10 轻量镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 先复制依赖文件（利用 Docker 缓存，加快构建）
COPY requirements.txt .

# 安装依赖（添加清华源加速）
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目所有文件
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]