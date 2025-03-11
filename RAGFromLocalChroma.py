import os
import torch
from langchain_chroma import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.embeddings.base import Embeddings
from typing import List
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.schema import Document


# 自定义 Qwen2 嵌入类
class Qwen2Embeddings(Embeddings):
    def __init__(self, model_path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.device = device

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 1  # 可以根据 GPU 内存调整
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                input_ids = self.tokenizer(batch_texts, return_tensors="pt", padding=True,
                                           truncation=True).input_ids.long().to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, output_hidden_states=True)
                    last_hidden_state = outputs.hidden_states[-1]
                    pooled_output = torch.mean(last_hidden_state, dim=1).squeeze()
                    if len(pooled_output.shape) == 1:
                        pooled_output = pooled_output.unsqueeze(0)
                    batch_embeddings = pooled_output.cpu().numpy().tolist()
                    for embedding in batch_embeddings:
                        print(f"Embedding dimension: {len(embedding)}")
                    embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"生成嵌入时出错: {e}")
                print(f"当前批次文本: {batch_texts}")
                # 打印当前 GPU 内存使用情况
                print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            finally:
                del input_ids, outputs, last_hidden_state, pooled_output
                torch.cuda.empty_cache()
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# 检查 GPU 是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

# 本地 Qwen2 模型路径
model_path = "G:/model/qwen/Qwen/Qwen2___5-1___5B-Instruct"

# 创建嵌入，使用自定义的 Qwen2 嵌入类
embeddings = Qwen2Embeddings(model_path, device)

# 本地 Chroma 数据库存储路径
persist_directory = 'G:\\chroma\\'

# 从本地加载 Chroma 向量数据库
try:
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
except Exception as e:
    print(f"加载本地 Chroma 数据库失败: {e}")
    raise

try:
    # 创建 Hugging Face 管道
    pipe = pipeline(
        "text-generation",
        model=AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device),
        tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True),
        temperature=0.7,
        num_return_sequences=1,
        device=device if device == "cuda" else -1,
        max_new_tokens=2048
    )
except Exception as e:
    print(f"模型加载失败: {e}")
    raise

# 创建 LangChain 的 LLM 实例
llm = HuggingFacePipeline(pipeline=pipe)

# 加载问答链
prompt_template = """根据以下文档内容回答问题：
{context}
问题: {question}
答案:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# 使用 create_stuff_documents_chain 替代 StuffDocumentsChain
chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
    document_variable_name="context"
)

# 用户问题
question = "西藏的人类起源？"

# 检索相关文档
try:
    similar_docs = db.similarity_search(question)
    print("检索到的相关文档:", similar_docs)  # 添加调试信息
except Exception as e:
    print(f"文档检索失败: {e}")
    raise

# 生成答案
try:
    # 'str' object has no attribute 'page_content'
    # context_text = [doc.page_content for doc in similar_docs]
    # answer = chain.invoke({"context": context_text, "question": question})
    context_docs = [Document(page_content=doc.page_content) for doc in similar_docs]
    answer = chain.invoke({"context": context_docs, "question": question})
    print("答案:", answer)
except Exception as e:
    print(f"答案生成失败: {e}")
    # 打印当前 GPU 内存使用情况
    print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
