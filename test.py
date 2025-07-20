# ✅ RAG 튜닝 대시보드 - Streamlit UI
import os
import uuid
import time
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# SQLite 이슈 해결
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# 환경 변수 로딩
api_key = st.secrets["UPSTAGE_API_KEY"]

# 디렉토리 설정
PERSIST_DIR = "./chroma_db"
PDF_SAVE_DIR = "./uploaded_pdfs"
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(PDF_SAVE_DIR, exist_ok=True)

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()

# 🔧 파라미터 튜닝 UI
st.sidebar.header("⚙️ RAG 파라미터 설정")
k_value = st.sidebar.slider("🔍 리트리버 k 값", 1, 10, 2)
chunk_size = st.sidebar.slider("📄 청크 사이즈", 100, 2000, 500, step=100)
chunk_overlap = st.sidebar.slider("🔁 청크 오버랩", 0, 500, 100, step=50)

# 📎 문서 업로드
st.sidebar.header("📎 문서 업로드")
uploaded_file = st.sidebar.file_uploader("PDF 또는 Word 파일 업로드", type=["pdf", "doc", "docx"])

if uploaded_file:
    file_path = os.path.join(PDF_SAVE_DIR, uploaded_file.name)
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "pdf":
            loader = PyPDFLoader(file_path)
        elif ext in ["doc", "docx"]:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            st.sidebar.error("❌ 지원하지 않는 파일 형식입니다.")
            st.stop()

        documents = loader.load()
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        docs = splitter.split_documents(documents)

        _ = Chroma.from_documents(
            docs,
            UpstageEmbeddings(model="solar-embedding-1-large"),
            persist_directory=PERSIST_DIR,
        )
        st.sidebar.success("✅ 문서 벡터 저장 완료!")
    else:
        st.sidebar.info("📂 이미 저장된 파일입니다.")

# 벡터스토어 로딩
vectorstore = Chroma(
    embedding_function=UpstageEmbeddings(model="solar-embedding-1-large"),
    persist_directory=PERSIST_DIR,
)
retriever = vectorstore.as_retriever(k=k_value)
chat = ChatUpstage(upstage_api_key=api_key)

# 리트리버 + QA 체인 구성
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "이전 대화가 있으면 질문을 독립적으로 재구성하세요. 답변하지 마세요."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
history_aware_retriever = create_history_aware_retriever(chat, retriever, contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 고든 램지 스타일의 요리 전문가입니다. 레시피 내용을 바탕으로 날카롭고 유쾌하게 답변하세요.\n"
     "최대 세 문장, 필요시 유머 사용, 모르면 모른다고 하세요.\n\n📍답변 내용:\n📍증거:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 🧠 챗봇 UI
st.title("🍳 요리 레시피 RAG 튜닝 챗봇")
st.markdown("예: '계란으로 만들 수 있는 요리 알려줘', '김치찌개는 어떻게 끓여요?'")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("요리에 대해 질문해보세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("작성 중..."):
            try:
                start_time = time.time()
                result = rag_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.messages,
                })
                elapsed = time.time() - start_time
                st.success(f"⏱️ 응답 시간: {elapsed:.2f}초 (k={k_value}, chunk={chunk_size}, overlap={chunk_overlap})")
                with st.expander("🔎 참고한 문서 보기"):
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
