import streamlit as st
from utils import qa_agent
from langchain.memory import ConversationBufferMemory

st.title("pdf阅读助手")

with st.sidebar:
    openai_api_key = st.text_input("请输入你的openai密钥",type="password")
    st.markdown("[获取openai密钥](https://baidu.com)")

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages = True,
        memory_key = "chat_history",
        output_key = "answer"
    )

upload_file = st.file_uploader("请上传pdf文件",type="pdf")
question = st.text_input("请输入你的问题",disabled=not upload_file)

if upload_file and question and not openai_api_key:
    st.info("请输入openai api密钥")

if upload_file and question and openai_api_key:
    with st.spinner("ai正在思考……"):
        response = qa_agent(openai_api_key,st.session_state["memory"],upload_file,question)

    st.write("### 答案")
    st.write(response["answer"])
    st.session_state["chat_history"] = response["chat_history"]

if "chat_history" in st.session_state:
    with st.expander("历史消息"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i+1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state["chat_history"])-2:
                st.divider()
