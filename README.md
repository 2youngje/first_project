# 🍳 요리 레시피 챗봇 (RAG 기반)

<div align="center">

![Python](https://img.shields.io/badge/python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-1.46.1-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/langchain-0.3.27-1C3C3C?style=flat)
![License](https://img.shields.io/badge/license-MIT-green?style=flat)

집에서 쉽게 만들어 먹을 수 있는 요리 레시피를 알려주는 **AI 챗봇**입니다.  
Upstage의 **Solar LLM**, **Chroma 벡터DB**, 그리고 **Streamlit**을 활용해 구현되었습니다.

</div>

---

## 📋 프로젝트 개요

이 프로젝트는 **RAG(Retrieval-Augmented Generation)** 기반의 요리 레시피 챗봇으로, 사용자가 업로드한 레시피 문서(PDF/Word)를 벡터 데이터베이스에 저장하고, 자연어 질의를 통해 관련 레시피 정보를 검색하여 답변을 생성합니다.

### 해결하려는 문제
- 요리 레시피 정보를 효율적으로 검색하고 활용하는 어려움
- 다양한 레시피 문서를 통합적으로 관리하고 질의응답하는 니즈

### 타겟 유저
- 요리 초보자부터 중급자까지
- 레시피 문서를 체계적으로 관리하고 싶은 사용자
- AI 기반 챗봇 기술에 관심이 있는 개발자

### 핵심 포인트
- **RAG 아키텍처**: 벡터 검색과 LLM을 결합한 정확한 답변 생성
- **대화 히스토리 인식**: 이전 대화 맥락을 고려한 질문 재구성
- **한국어 감성 분석**: KoELECTRA 기반 사용자 질문 감성 분류
- **고든 램지 스타일**: 유쾌하고 직설적인 답변 톤

---

## ✨ 주요 기능

- [x] **PDF/Word 문서 업로드 및 벡터화**
  - 레시피 문서를 업로드하여 Chroma 벡터DB에 자동 저장
  - Upstage Solar Embedding 모델을 활용한 고품질 임베딩

- [x] **자연어 질의응답 (RAG)**
  - "김치볶음밥 만드는 법 알려줘", "두부로 만들 수 있는 요리 있어?" 등 자연어 질문 지원
  - 관련 레시피 문서를 자동 검색하여 정확한 답변 생성

- [x] **대화 히스토리 인식**
  - 이전 대화 맥락을 고려한 질문 재구성 (History-Aware Retriever)
  - 연속적인 대화에서도 정확한 답변 제공

- [x] **한국어 감성 분석**
  - KoELECTRA 기반 사용자 질문의 긍정/부정 감성 분류
  - 실시간 감성 분석 결과 표시

- [x] **Streamlit 기반 웹 UI**
  - 직관적인 채팅 인터페이스
  - 참고 문서 확인 기능
  - 단어 단위 스트리밍 답변 출력

---

## 🛠️ 기술 스택

### 언어 및 프레임워크
- **Python 3.11+**
- **Streamlit 1.46.1** - 웹 UI 프레임워크
- **LangChain 0.3.27** - RAG 체인 구성 및 문서 처리

### AI/ML 라이브러리
- **Upstage Solar LLM** - 대규모 언어 모델 (질문 재구성 및 답변 생성)
- **Upstage Solar Embedding** - 텍스트 임베딩 모델
- **Chroma 0.2.4** - 벡터 데이터베이스
- **Transformers 4.44.2** - KoELECTRA 감성 분석 모델
- **PyTorch 2.7.1** - 딥러닝 프레임워크

### 문서 처리
- **PyPDFLoader** - PDF 파일 로딩
- **UnstructuredWordDocumentLoader** - Word 파일 로딩
- **pysqlite3-binary** - SQLite 호환성 패키지

### 기타 도구
- **python-dotenv** - 환경 변수 관리
- **LangChain Community** - 문서 로더 및 유틸리티

---

## 🚀 설치 및 실행 방법

### 선행 조건

- **Python 3.11 이상**
- **Git** (저장소 클론용)
- **Upstage API 키** - [Upstage Console](https://console.upstage.ai)에서 발급

### 설치 단계

1. **저장소 클론**
   ```bash
   git clone <repository-url>
   cd first_project
   ```

2. **가상환경 생성 및 활성화** (권장)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 또는
   venv\Scripts\activate  # Windows
   ```

3. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

4. **환경 변수 설정**

   Streamlit secrets를 사용하는 경우:
   - `.streamlit/secrets.toml` 파일 생성
   ```toml
   UPSTAGE_API_KEY = "your_upstage_api_key_here"
   ```

   또는 `.env` 파일 사용:
   ```env
   UPSTAGE_API_KEY=your_upstage_api_key_here
   ```

### 실행 방법

**메인 앱 실행:**
```bash
streamlit run recipe_bot.py
```

브라우저에서 `http://localhost:8501`로 접속하여 사용할 수 있습니다.

**RAG 튜닝 대시보드 실행:**
```bash
streamlit run test.py
```

---

## 💡 사용 예시

### 기본 사용 플로우

1. **레시피 문서 업로드**
   - 사이드바에서 PDF 또는 Word 파일 업로드
   - 자동으로 벡터DB에 저장됨

2. **질문 입력**
   - 채팅 입력창에 요리 관련 질문 입력
   - 예: "김치볶음밥 만드는 법 알려줘"

3. **답변 확인**
   - 고든 램지 스타일의 유쾌한 답변 수신
   - 참고한 레시피 문서 확인 가능
   - 질문의 감성 분석 결과 확인

### 예시 질의

```
사용자: "계란으로 만들 수 있는 요리 알려줘"
봇: "계란으로는 계란말이, 스크램블 에그, 계란국 등이 있어요. 
     간단하고 영양 만점이니 꼭 시도해보세요!"

사용자: "김치찌개는 어떻게 끓여요?"
봇: "김치찌개는 돼지고기와 김치를 볶은 뒤 물을 넣고 끓이면 돼요. 
     고춧가루와 대파를 넣어 마무리하면 완성!"
```

---

## 📁 프로젝트 구조

```
first_project/
├── recipe_bot.py          # 메인 챗봇 애플리케이션
├── test.py                # RAG 튜닝 대시보드 (파라미터 조정용)
├── requirements.txt       # Python 패키지 의존성 목록
├── README.md             # 프로젝트 문서
├── chroma_db/            # Chroma 벡터 데이터베이스 저장 디렉토리
│   └── ...               # 벡터 인덱스 파일들
└── uploaded_pdfs/        # 업로드된 레시피 문서 저장 디렉토리
    └── ...               # PDF/Word 파일들
```

### 주요 파일 설명

- **`recipe_bot.py`**: 메인 챗봇 애플리케이션
  - Streamlit UI 구성
  - RAG 체인 설정 (History-Aware Retriever + QA Chain)
  - 감성 분석 통합
  - 문서 업로드 및 벡터화 처리

- **`test.py`**: RAG 파라미터 튜닝용 대시보드
  - k 값, chunk size, chunk overlap 조정 가능
  - 응답 시간 측정 기능

---

## 🤖 AI/ML 프로젝트 정보

### 사용한 모델

- **Upstage Solar LLM**: 질문 재구성 및 답변 생성
- **Upstage Solar Embedding (solar-embedding-1-large)**: 텍스트 임베딩
- **monologg/koelectra-base-v3-discriminator**: 한국어 감성 분석

### RAG 아키텍처

1. **문서 로딩**: PDF/Word 파일을 LangChain Document로 변환
2. **임베딩 생성**: Solar Embedding 모델로 벡터화
3. **벡터 저장**: Chroma DB에 영구 저장 (멀티턴 지원)
4. **검색 (Retrieval)**: 사용자 질문과 유사한 문서 청크 검색 (k=2)
5. **질문 재구성**: 이전 대화 히스토리를 고려한 독립 질문 생성
6. **답변 생성**: 검색된 문서를 컨텍스트로 활용한 LLM 답변 생성

### 프롬프트 엔지니어링

- **시스템 프롬프트**: "고든 램지 스타일의 요리 전문가" 페르소나
- **답변 스타일**: 날카롭고 직설적이지만 유쾌한 톤, 최대 3문장
- **폴백 처리**: 모를 경우 솔직하게 "모른다"고 답변

---

## 🗺️ 향후 개선 사항

### 기능 개선
- [ ] **웹 크롤링 기반 레시피 수집**
  - 문서 기반이 아닌 실시간 웹 크롤링으로 레시피 정보 확장

- [ ] **재료 기반 레시피 추천**
  - 사용자가 보유한 재료를 입력받아 가능한 요리 추천

- [ ] **모듈화 및 코드 리팩토링**
  - 기능별 모듈 분리 (RAG 체인, 감성 분석, 문서 처리 등)
  - 설정 파일 분리

### 성능 개선
- [ ] **응답 속도 최적화**
  - 캐싱 전략 도입
  - 비동기 처리 적용

- [ ] **검색 정확도 향상**
  - 하이브리드 검색 (키워드 + 벡터)
  - 리랭킹 모델 적용

### 배포 및 인프라
- [ ] **Docker 컨테이너화**
  - 배포 편의성을 위한 Docker 이미지 생성

- [ ] **Streamlit Cloud 배포**
  - 클라우드 환경에서 서비스 제공

- [ ] **모니터링 및 로깅**
  - 사용자 질문/답변 로깅
  - 성능 메트릭 수집

---

## 📝 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

---

## 📧 문의

프로젝트에 대한 문의사항이나 제안사항이 있으시면 이슈를 등록해주세요.

---

<div align="center">

**Made with ❤️ using Streamlit, LangChain, and Upstage Solar**

</div>
