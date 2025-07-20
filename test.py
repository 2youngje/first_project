import os
import uuid
import time
import tempfile
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# SQLite 오류 회피 (Chroma with pysqlite3)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# API 키 로딩
api_key = st.secrets["UPSTAGE_API_KEY"]

# 디렉토리 생성
PERSIST_DIR = "./chroma_db"
PDF_SAVE_DIR = "./uploaded_pdfs"
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(PDF_SAVE_DIR, exist_ok=True)

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()



# 📎 파일 업로드
st.sidebar.header("📎 레시피 문서 업로드")
uploaded_file = st.sidebar.file_uploader("PDF 또는 Word 파일을 업로드하세요", type=["pdf", "doc", "docx"])

if uploaded_file:
    file_path = os.path.join(PDF_SAVE_DIR, uploaded_file.name)

    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.sidebar.success("✅ 파일 저장 완료!")

        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "pdf":
            loader = PyPDFLoader(file_path)
        elif ext in ["doc", "docx"]:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            st.sidebar.error("❌ 지원하지 않는 파일 형식입니다.")
            st.stop()

        # 🔀 문서 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        documents = loader.load()
        split_docs = text_splitter.split_documents(documents)

        # 💾 Chroma 저장
        _ = Chroma.from_documents(
            split_docs,
            UpstageEmbeddings(model="solar-embedding-1-large"),
            persist_directory=PERSIST_DIR,
        )
        st.sidebar.success("✅ 벡터 저장 완료!")
    else:
        st.sidebar.info("📂 이미 저장된 파일입니다.")

# ⚙️ 하이퍼파라미터 슬라이더
st.sidebar.header("🧪 성능 테스트용 설정")
k_value = st.sidebar.slider("🔍 리트리버 Top-K", 1, 10, 2)
chunk_size = st.sidebar.slider("🧩 Chunk Size", 100, 2000, 500, step=100)
chunk_overlap = st.sidebar.slider("🔄 Chunk Overlap", 0, 500, 50, step=50)

# 🔄 항상 로딩
vectorstore = Chroma(
    embedding_function=UpstageEmbeddings(model="solar-embedding-1-large"),
    persist_directory=PERSIST_DIR,
)
retriever = vectorstore.as_retriever(k=k_value)
chat = ChatUpstage(upstage_api_key=api_key)

# 🧠 리트리버 체인
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "사용자의 요리 관련 질문이 이전 대화와 관련이 있으면, 독립적인 질문으로 다시 구성하세요. 답변은 하지 마세요."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
history_aware_retriever = create_history_aware_retriever(chat, retriever, contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 욕쟁이 할머니 입니다. "
     "그런걸 물어 보냐는 듯이 비아냥거려"
     "모르면 모른다고 솔직하게 말하는게 아닌 거짓말을 해봐.\n\n"
     "📍답변 내용:\n📍증거:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 💬 Streamlit UI
st.title("🍳 집에서 만들어 먹는 요리 레시피 챗봇")
st.markdown("예: ‘김치볶음밥 만드는 법 알려줘’, ‘두부로 만들 수 있는 요리 있어?’")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("요리에 대해 궁금한 걸 물어보세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("작성 중..."):
            try:
                result = rag_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.messages,
                })

                with st.expander("🔎 참고한 레시피 문서 보기"):
                    st.write(result["context"])

                for chunk in result["answer"].split(" "):
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.08)
                message_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"❌ 오류 발생: {str(e)}"
                message_placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
