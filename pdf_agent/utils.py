from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import time  # 导入 time 模块

def qa_agent(openai_api_key, memory, upload_file, question):
    model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key, openai_api_base="https://api.aigc369.com/v1")
    file_content = upload_file.read()
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)
    
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 减小块大小
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)
    
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key, openai_api_base="https://api.aigc369.com/v1")
    
    # 批量处理
    all_embeddings = []
    batch_size = 10  # 每次处理10个文本块
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        for _ in range(3):  # 最多重试3次
            try:
                batch_embeddings = embeddings_model.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                print(f"Error: {e}, retrying...")
                time.sleep(2)  # 等待2秒后重试
    
    db = FAISS.from_documents(texts, all_embeddings)
    retriever = db.as_retriever()
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )
    response = qa.invoke({"chat_history": memory, "question": question})
    return response
