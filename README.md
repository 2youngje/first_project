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
├── requirements.txt  # 파이썬 라이브러리 설치 목록
├── .env              # API 키 저장용 (개별 생성) -> .gitignore로 github에는 안올라감 
└── README.md         # 프로젝트 소개 및 개발 현황
```

## 🚀 실행 방법

1. **필요 라이브러리 설치**

```bash
pip install -r requirements.txt

```
2. **.env 파일 생성**

```env
UPSTAGE_API_KEY=your_upstage_api_key_here
```
3. **앱 실행**

```bash
streamlit run app.py
```

4. **레시피 .pdf 추가**
```
레시피 PDF를 추가해 챗봇의 성능을 올려주세요
```

## 📌 TODO (추후 개선 예정)

- ✅ **레시피 데이터 외부 PDF 에서 불러오기**  
  현재는 코드에 레시피를 정의하고 있음 -> 외부 파일로 분리, 추가 할 수 있게 만들기 -> pdf를 추가 해야지 작동!

- ✅ **레시피 추천 기능**  
  사용자의 재료 입력이나 이전 질문 기반으로 요리 추천  
  예: “냉장고에 두부, 파 있어요” -> 가능한 요리 제안

- ✅ **Streamlit UX 개선**  
  답변 시 로딩 "생각하는 중" -> spiner와 같은 기능 추가

- ✅ **LLM 호출 에러 처리 및 안내 메시지 개선**  
  API KEY 인증실패 시, 답변 생성이 안됐을 시 친절한 안내 제공

- ✅ **Chroma 벡터 DB를 디스크 저장소로 전환**  
  매번 초기화하지 않고, 저장된 벡터를 재사용(멀티턴) -> 완료

- ❌ **사용자 질문 기록 저장**
  redis DB를 사용해서 기록 -> streamlit 배포로 취소

- [ ] **모듈화**
  코드 정리(우선 동작하게 만들기) -> 이후 고도화

- [ ] **PDF가 아닌 웹 크롤링으로 동작**
  이후 고도화 작업에서 필요
  
- [ ] **성능평가 지표**