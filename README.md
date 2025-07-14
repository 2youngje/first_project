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
# 🧠 Programming Syntax Assistant (프로그래밍 문법 도우미)

> LLM + RAG 기반으로 다양한 프로그래밍 문법을 친절하게 설명해주는 문법 도우미 웹 앱입니다.

---

## 📌 프로젝트 개요

이 프로젝트는 사용자의 질문에 대해 Hugging Face 데이터셋을 바탕으로 문법 자료를 검색하고, LLM을 통해 자연어로 설명해주는 RAG 기반 문법 도우미입니다.

- ✅ LLM + FAISS 기반 문서 검색 (RAG)
- ✅ 다양한 프로그래밍 언어 문법 지원 (Python 중심)
- ✅ 코드 예시 + 개념 설명
- ✅ 간단한 UI (Streamlit)

---

## 🏗️ 기술 스택

| 항목       | 내용 |
|------------|------|
| 언어       | Python 3.11.8 |
| 백엔드     | OpenAI API or Hugging Face Transformers |
| 검색엔진   | FAISS (Facebook AI Similarity Search) |
| 데이터     | Hugging Face Datasets (`code_search_net`, `mbpp`) |
| 프론트엔드 | Streamlit |

---

## 📁 폴더 구조

```
project-root/
├── app.py # Streamlit 메인 앱
├── data/ # 문법 데이터셋 (전처리 후 jsonl 등)
├── index/ # FAISS 벡터 인덱스 저장
├── rag_utils.py # 검색 및 응답 생성 함수
├── requirements.txt
└── README.md
```
