"""构建知识库程序"""
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os
# 自定义支持 UTF-8 的加载器
class UTF8TextLoader(TextLoader):
    def __init__(self, file_path, encoding="utf-8"):
        super().__init__(file_path, encoding=encoding)

# 1. 加载文档（支持.txt，可扩展.md等）
loader = DirectoryLoader("./knowledge_base", glob="**/*.txt", loader_cls=UTF8TextLoader)
docs = loader.load()

# 2. 分割文本（每段500字符，重叠50）
splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
chunks = splitter.split_documents(docs)

# 3. 生成向量并存储（需要embedding模型，先pull 一个轻量模型）
# 执行：ollama pull nomic-embed-text
embeddings = OllamaEmbeddings(model = "nomic-embed-text")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

print(f"知识库构建完成，共{len(chunks)} 个片段。")