<div align=center>
<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=200&section=header&text=first_project&fontSize=90" />
</div>
	<div align=center>
		<h2>🌱 welcome to my github 🌱</h2>
		<h3>📚 Tech Stack 📚</h3>
		<p>✨ Platforms & Languages ✨</p>
	</div>

<div align="center">
	<img src="https://img.shields.io/badge/python-007396?style=flat&logo=python&logoColor=white" />
</div>

# 🍳 요리 레시피 챗봇 (RAG 기반)

집에서 쉽게 만들어 먹을 수 있는 요리 레시피를 알려주는 **AI 챗봇**입니다.  
Upstage의 **Solar LLM**, **Chroma 벡터DB**, 그리고 **Streamlit**을 활용해 구현되었습니다.

---

## 🧠 주요 기능

- ✅ 요리에 대한 질문에 대답
- ✅ 문맥 기반 질문 재구성 (RAG + 히스토리 리트리버)
- ✅ Streamlit UI
- ✅ Solar LLM API 기반 답변 생성

---

## 🛠️ 사용 기술

| 도구 | 설명 |
|------|------|
| [Upstage Solar LLM](https://docs.upstage.ai) | 질문 요약 + 답변 생성 |
| [LangChain](https://docs.langchain.com) | RAG 체인 구성 |
| [Chroma](https://docs.trychroma.com) | 벡터 DB |
| [Streamlit](https://streamlit.io) | 웹 프론트엔드 UI |
| Python | 전체 앱 구현 |
| dotenv | 환경 변수 관리 |

---

```
.
├── recipe_bot.py     # 메인 코드
├── requirements.txt  # 의존성 목록
├── .env              # API 키 저장용 (개별 생성) -> .gitignore로 github에는 안올라감 
└── README.md         # 프로젝트 소개 및 개발 현황
```