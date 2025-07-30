import os
import uuid
import time
import sys
import re

import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from transformers import pipeline
import torch

# SQLite 패키지 이슈 해결
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# API 키 로딩 (Streamlit secrets 또는 .env)
api_key = st.secrets.get("UPSTAGE_API_KEY", "")

# --- 한국어 감성 분석 파이프라인 로딩 ---
@st.cache_resource
def load_sentiment_pipeline():
    # KoELECTRA 감성 분석 (LABEL_0=부정, LABEL_1=긍정)
    return pipeline("sentiment-analysis", model="monologg/koelectra-base-v3-discriminator")

sentiment_pipeline = load_sentiment_pipeline()

def preprocess_text(text: str) -> str:
    # 기본 전처리: 특수문자, 중복공백 제거 등
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def classify_sentiment(text: str) -> str:
    clean_text = preprocess_text(text)
    if not clean_text:
        return "**감성 분석 결과:** 분석 불가 (빈 문장)"
    result = sentiment_pipeline(clean_text)[0]
    label = result['label']
    score = result['score']
    label_kr = "긍정적" if label == "LABEL_1" else "부정적"
    emoji = "👍" if label == "LABEL_1" else "👎"
    return f"**감성 분석 결과:** {label_kr} ({score:.2f}) {emoji}"

# --- 벡터 저장소 및 파일 저장 위치 ---
PERSIST_DIR = "./chroma_db"
PDF_SAVE_DIR = "./uploaded_pdfs"
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(PDF_SAVE_DIR, exist_ok=True)

# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()

# --- 사이드바: 파일 업로드 ---
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

        documents = loader.load()
        _ = Chroma.from_documents(
            documents,
            UpstageEmbeddings(model="solar-embedding-1-large"),
            persist_directory=PERSIST_DIR,
        )
        st.sidebar.success("✅ 레시피 벡터스토어에 저장 완료!")
    else:
        st.sidebar.info("📂 이미 저장된 파일입니다.")

# --- 벡터스토어 로딩 ---
vectorstore = Chroma(
    embedding_function=UpstageEmbeddings(model="solar-embedding-1-large"),
    persist_directory=PERSIST_DIR,
)
retriever = vectorstore.as_retriever(k=2)

# --- Solar LLM 초기화 ---
chat = ChatUpstage(upstage_api_key=api_key)

# --- 질문 리프레이징 체인 ---
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "사용자의 요리 관련 질문이 이전 대화와 관련이 있으면, 독립적인 질문으로 다시 구성하세요. 답변은 하지 마세요."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
history_aware_retriever = create_history_aware_retriever(chat, retriever, contextualize_q_prompt)

# --- RAG QA 체인 프롬프트 ---
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

# --- UI: 타이틀과 안내 ---
st.title("🍳 집에서 만들어 먹는 요리 레시피 챗봇")
st.markdown("예: ‘김치볶음밥 만드는 법 알려줘’, ‘두부로 만들 수 있는 요리 있어?’")

# --- 이전 대화 출력 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 사용자 질문 입력 처리 ---
if prompt := st.chat_input("요리에 대해 궁금한 걸 물어보세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    sentiment_result = classify_sentiment(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("작성 중..."):
            try:
                # RAG 답변 생성
                result = rag_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.messages,
                })

                # 참고 문서 보기 확장 영역
                with st.expander("🔎 참고한 레시피 문서 보기"):
                    st.write(result["context"])

                # 답변을 단어 단위로 애니메이션 출력
                for chunk in result["answer"].split(" "):
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.05)  # 속도 조절 가능
                message_placeholder.markdown(full_response)

                st.markdown("---")
                st.markdown(sentiment_result)

            except Exception as e:
                full_response = f"❌ 오류 발생: {str(e)}"
                message_placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
