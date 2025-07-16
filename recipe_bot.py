import os
import uuid
from typing import List

import streamlit as st
from dotenv import load_dotenv

# LangChain 및 Chroma 관련 모듈
from langchain_core.documents import Document
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.document_loaders import PyPDFLoader
import tempfile

# .env 파일에서 API 키 불러오기
load_dotenv()
api_key = os.getenv("UPSTAGE_API_KEY")

# 세션 초기화 (Streamlit이 상태 기억하도록 설정)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    
# 사이드바: PDF 업로드
st.sidebar.header("📎 레시피 PDF 업로드")
uploaded_file = st.sidebar.file_uploader("PDF 파일을 업로드하세요", type="pdf")

# PDF 문서 -> List[Document] 변환
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    recipes: List[Document] = loader.load()  # ✅ 이 줄이 핵심: 업로드된 PDF가 recipes가 됨
    st.sidebar.success("✅ PDF에서 레시피 로딩 완료!")

    # 3. 벡터스토어 생성
    vectorstore = Chroma.from_documents(recipes, UpstageEmbeddings(model="solar-embedding-1-large"))
    retriever = vectorstore.as_retriever(k=2)

    # 이하 그대로 RAG 체인 구성, 채팅 로직 등 사용 가능
    ...
else:
    st.warning("👈 좌측 사이드바에서 PDF 파일을 업로드해주세요!")
    st.stop()

# 앱 제목 및 안내 문구 출력
st.title("🍳 집에서 만들어 먹는 요리 레시피 챗봇")
st.markdown("예: ‘김치볶음밥 만드는 법 알려줘’, ‘두부로 만들 수 있는 요리 있어?’")

# 레시피 문서 정의 (고정된 문서들) -> 이 기능을 pdf를 가져와서 실행하는 고도화를 진행해야한다....
# recipes: List[Document] = [
#     Document(page_content="김치볶음밥: 밥, 김치, 참기름, 간장, 설탕, 대파, 계란을 사용해 볶음밥을 만든다."),
#     Document(page_content="된장찌개: 된장, 두부, 애호박, 양파, 고추, 마늘 등을 넣고 끓인다."),
#     Document(page_content="계란말이: 계란, 당근, 파, 소금을 넣고 얇게 부쳐 돌돌 만다."),
#     Document(page_content="부침개: 부침가루, 물, 야채(부추, 양파 등), 소금을 넣고 팬에 부쳐 완성."),
#     Document(page_content="떡볶이: 떡, 고추장, 고춧가루, 어묵, 양파, 설탕을 넣고 자작하게 끓인다."),
# ]
# 이 부분은 사이드 바가 되는 것이다.

# 문서를 임베딩하여 벡터스토어 생성 (Chroma + Solar 임베딩)
vectorstore = Chroma.from_documents(recipes, UpstageEmbeddings(model="solar-embedding-1-large"))

# 검색 리트리버 구성
retriever = vectorstore.as_retriever(k=2)

# Solar 기반 챗봇 객체 생성
chat = ChatUpstage(upstage_api_key=api_key)

# 질문 재구성용 시스템 프롬프트 정의
contextualize_q_system_prompt = """사용자의 요리 관련 질문이 이전 대화와 관련이 있으면, 독립적인 질문으로 다시 구성하세요. 답변은 하지 마세요."""

# 대화 히스토리를 반영한 리트리버 구성
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
history_aware_retriever = create_history_aware_retriever(chat, retriever, contextualize_q_prompt)

# 요리 전문 어시스턴트 역할의 시스템 프롬프트
qa_system_prompt = """요리 전문가 어시스턴트입니다. 질문에 답하기 위해 제공된 레시피 내용을 활용하세요. 모르면 모른다고 말하고, 답변은 세 문장 이내로 간결하게 작성하세요.
📍답변 내용:
📍증거:
{context}
"""

# 문서와 함께 답변 생성할 수 있도록 프롬프트 구성
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# 문서 기반 답변 생성 체인 구성 (RAG)
question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 기존 대화 메시지를 UI에 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 질문 입력 처리
if prompt := st.chat_input("요리에 대해 궁금한 걸 물어보세요!"):
    # 입력된 질문을 세션 상태에 저장
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 사용자 질문 출력
    with st.chat_message("user"):
        st.markdown(prompt)

    # 어시스턴트 응답 생성 및 출력
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # RAG 체인 호출
        result = rag_chain.invoke({
            "input": prompt,
            "chat_history": st.session_state.messages,
        })

        # 검색된 문서 내용 펼쳐보기
        with st.expander("🔎 참고한 레시피 문서 보기"):
            st.write(result["context"])

        # 응답을 한 단어씩 출력 (타자 치는 효과)
        for chunk in result["answer"].split(" "):
            full_response += chunk + " "
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # 어시스턴트 답변 저장
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # 주석 추가