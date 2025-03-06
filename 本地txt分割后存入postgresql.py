import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import chardet
import psycopg2
import jieba

# 检测文件编码
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

# 加载本地文本文件
def load_documents(data_folder):
    documents = []
    encodings_to_try = ['GB2312', 'GBK', 'GB18030']
    failed_files = []
    for filename in os.listdir(data_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_folder, filename)
            detected_encoding = detect_encoding(file_path)
            for encoding in [detected_encoding] + encodings_to_try:
                try:
                    loader = TextLoader(file_path, encoding=encoding)
                    docs = loader.load()
                    documents.extend(docs)
                    break
                except Exception as e:
                    if encoding == encodings_to_try[-1]:
                        print(f"加载文件 {file_path} 失败: {e}，所有编码尝试均失败。")
                        failed_files.append(file_path)
    if failed_files:
        print("以下文件加载失败:", failed_files)
    return documents

# 加载文档
data_folder = "G:\\world"
documents = load_documents(data_folder)
# 方法一
# text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

# 方法一结束
# 方法二
# 合并所有文档内容为一个字符串
all_content = "".join([doc.page_content for doc in documents])

# 使用 jieba 分词
words = jieba.lcut(all_content)

# 按句号进行句子分割
sentence_segments = []
current_sentence = ""
for word in words:
    current_sentence += word.strip().strip("上一页 目录页 下一页")
    if "。" in word:
        sentence_segments.append(current_sentence)
        current_sentence = ""

# 处理最后一个可能没有句号的句子
if current_sentence:
    sentence_segments.append(current_sentence)

# 连接到 PostgreSQL 数据库
try:
    conn = psycopg2.connect(
        database="postgres",
        user="postgres",
        password="123456",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()

    # 检查 xizang 表是否存在，不存在则创建
    cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'xizang');")
    table_exists = cursor.fetchone()[0]
    if not table_exists:
        create_table_query = "CREATE TABLE xizang (id SERIAL PRIMARY KEY, content TEXT, type VARCHAR(255));"
        cursor.execute(create_table_query)
        conn.commit()

    # 普通插入方法
    for segment in sentence_segments:
        insert_query = "INSERT INTO xizang (content, type) VALUES (%s, %s);"
        cursor.execute(insert_query, (segment, '西藏'))
    conn.commit()
    print("分割后的文本已成功存储到 PostgreSQL")
    # 方法一
    # 普通插入方法
    """
    for doc in docs:
        insert_query = "INSERT INTO xizang (content, type) VALUES (%s, %s);"
        cursor.execute(insert_query, (doc.page_content, '西藏'))
    conn.commit()
    print("分割后的文本已成功存储到 PostgreSQL")
    """
    #方法一结束

except (Exception, psycopg2.Error) as error:
    print(f"存储到 PostgreSQL 时出错: {error}")
finally:
    if conn:
        cursor.close()
        conn.close()
        print("PostgreSQL 连接已关闭")
# 方法二结束