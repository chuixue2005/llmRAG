import os
import shutil
import torch
from sqlalchemy import create_engine, text
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM


# 自定义 Qwen2 嵌入类
class Qwen2Embeddings(Embeddings):
    def __init__(self, model_path, device):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
            self.device = device
        except Exception as e:
            print(f"模型加载出错: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 16  # 可以根据 GPU 内存调整
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_input_ids = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).input_ids.long().to(self.device)
            try:
                with torch.no_grad():
                    outputs = self.model(batch_input_ids, output_hidden_states=True)
                    last_hidden_state = outputs.hidden_states[-1]
                    pooled_output = torch.mean(last_hidden_state, dim=1).squeeze()
                    batch_embeddings = pooled_output.cpu().numpy().tolist()
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"生成嵌入时出错: {e}")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# 配置 PostgreSQL 数据库连接
DB_USER = ""
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "postgres"

# 创建数据库引擎
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# 从 PostgreSQL 中查询数据
def query_data_from_postgres():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT content FROM xizang"))
            texts = [row[0] for row in result]
        return texts
    except Exception as e:
        print(f"数据库查询出错: {e}")
        return []

# 清空本地路径
def clear_local_path(persist_directory):
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    os.makedirs(persist_directory, exist_ok=True)

# 检查 GPU 是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

# 加载数据
texts = query_data_from_postgres()

if not texts:
    print("未获取到有效数据，程序退出。")
else:
    # 把文本数据转换为 Document 对象列表
    document_objects = [Document(page_content=text) for text in texts]

    # 本地 Qwen2 模型路径
    model_path = "G:/model/qwen/Qwen/Qwen2___5-1___5B-Instruct"

    # 创建嵌入，使用自定义的 Qwen2 嵌入类
    embeddings = Qwen2Embeddings(model_path, device)

    # 本地存储路径
    persist_directory = 'G:\\chroma\\'

    # 清空本地路径
    clear_local_path(persist_directory)

    # 创建 Chroma 向量数据库并保存到本地
    db = Chroma.from_documents(documents=document_objects, embedding=embeddings, persist_directory=persist_directory)

    # 持久化保存
    # db.persist() # 已过时
    print("数据已成功保存到本地 Chroma 数据库。")
