import os
import uuid
from typing import List

import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("UPSTAGE_API_KEY")

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()

# 앱 제목
st.title("🍳요리 레시피 챗봇")
st.markdown("예: ‘김치볶음밥 만드는 법 알려줘’, ‘두부로 만들 수 있는 요리 있어?’")

# 레시피 데이터 정의
recipes: List[Document] = [
    Document(page_content="김치볶음밥: 밥, 김치, 참기름, 간장, 설탕, 대파, 계란을 사용해 볶음밥을 만든다."),
    Document(page_content="된장찌개: 된장, 두부, 애호박, 양파, 고추, 마늘 등을 넣고 끓인다."),
    Document(page_content="계란말이: 계란, 당근, 파, 소금을 넣고 얇게 부쳐 돌돌 만다."),
    Document(page_content="부침개: 부침가루, 물, 야채(부추, 양파 등), 소금을 넣고 팬에 부쳐 완성."),
    Document(page_content="떡볶이: 떡, 고추장, 고춧가루, 어묵, 양파, 설탕을 넣고 자작하게 끓인다."),
]

# 벡터스토어 및 리트리버 생성
vectorstore = Chroma.from_documents(recipes, UpstageEmbeddings(model="solar-embedding-1-large"))
retriever = vectorstore.as_retriever(k=2)
chat = ChatUpstage(upstage_api_key=api_key)

# 프롬프트 정의
contextualize_q_system_prompt = """사용자의 요리 관련 질문이 이전 대화와 관련이 있으면, 독립적인 질문으로 다시 구성하세요. 답변은 하지 마세요."""
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
history_aware_retriever = create_history_aware_retriever(chat, retriever, contextualize_q_prompt)

qa_system_prompt = """요리 전문가 어시스턴트입니다. 질문에 답하기 위해 제공된 레시피 내용을 활용하세요. 모르면 모른다고 말하고, 답변은 세 문장 이내로 간결하게 작성하세요.
📍답변 내용:
📍증거:
{context}
"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 기존 메세지 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 채팅 입력
if prompt := st.chat_input("요리에 대해 궁금한 걸 물어보세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        result = rag_chain.invoke({
            "input": prompt,
            "chat_history": st.session_state.messages,
        })

        with st.expander("🔎 참고한 레시피 문서 보기"):
            st.write(result["context"])

        for chunk in result["answer"].split(" "):
            full_response += chunk + " "
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# 초기 : 사이드 바 제거(메인 화면에서 구동 할 수 있게)