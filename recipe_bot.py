import os
import tempfile
import uuid
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
import time

# SQLite 패키지 이슈 해결
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Streamlit secrets에서 API 키 로딩
api_key = st.secrets["UPSTAGE_API_KEY"]

# 벡터 저장소 및 PDF 업로드 디렉토리 설정
PERSIST_DIR = "./chroma_db"
PDF_SAVE_DIR = "./uploaded_pdfs"
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(PDF_SAVE_DIR, exist_ok=True)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()

# 사이드바 - PDF 또는 Word 업로드
st.sidebar.header("📎 레시피 문서 업로드")
uploaded_file = st.sidebar.file_uploader("PDF 또는 Word 파일을 업로드하세요", type=["pdf", "doc", "docx"])

# 업로드된 파일 처리
if uploaded_file:
    file_path = os.path.join(PDF_SAVE_DIR, uploaded_file.name)

    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.sidebar.success("✅ 파일 저장 완료!")

        # 파일 확장자 확인 후 적절한 로더 선택
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "pdf":
            loader = PyPDFLoader(file_path)
        elif ext in ["doc", "docx"]:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            st.sidebar.error("❌ 지원하지 않는 파일 형식입니다.")
            st.stop()

        # 문서 벡터화 및 저장
        recipes = loader.load()
        _ = Chroma.from_documents(
            recipes,
            UpstageEmbeddings(model="solar-embedding-1-large"),
            persist_directory=PERSIST_DIR,
        )
        st.sidebar.success("✅ 레시피 벡터스토어에 저장 완료!")
    else:
        st.sidebar.info("📂 이미 저장된 파일입니다.")

# 항상 벡터스토어 로딩
vectorstore = Chroma(
    embedding_function=UpstageEmbeddings(model="solar-embedding-1-large"),
    persist_directory=PERSIST_DIR,
)
retriever = vectorstore.as_retriever(k=2)

# Solar LLM 초기화
chat = ChatUpstage(upstage_api_key=api_key)

# 과거 대화 기반 질문 리프레이징 체인
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "사용자의 요리 관련 질문이 이전 대화와 관련이 있으면, 독립적인 질문으로 다시 구성하세요. 답변은 하지 마세요."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
history_aware_retriever = create_history_aware_retriever(chat, retriever, contextualize_q_prompt)

# RAG QA 체인 프롬프트
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 고든 램지 스타일의 요리 전문가입니다. "
     "질문에 답하기 위해 제공된 레시피 내용을 활용하세요. "
     "답변은 날카롭고 직설적이지만 유쾌하게, 최대 세 문장으로 간결하게 작성하세요. "
     "필요 시, 유머와 고든 램지 특유의 표현을 섞어 사용하세요. "
     "모르면 모른다고 솔직하게 말하세요.\n\n"
     "📍답변 내용:\n📍증거:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 메인 UI 구성
st.title("🍳 집에서 만들어 먹는 요리 레시피 챗봇")
st.markdown("예: ‘김치볶음밥 만드는 법 알려줘’, ‘두부로 만들 수 있는 요리 있어?’")

# 과거 대화 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 질문 처리
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
