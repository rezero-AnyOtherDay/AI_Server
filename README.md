# AI Agent 기반의 보호자용 진단 서비스

음성 데이터를 분석하여 피보호자가 겪는 뇌질환의 징후를 조기에 발견하고, RAG 기반의 의학적 판단을 통해 맞춤형 리포트를 보호자에게 제공하는 AI 서버입니다.

## Project Overview

본 프로젝트는 FastAPI 서버를 기반으로 작동하며, whisper-tiny 모델로 피보호자의 음성을 분석하여 질환을 분류하고, AI Agent가 RAG를 통해 의학 문서를 참조하여 종합적인 진단 리포트를 도출하도록 합니다.

### 사용 tool

* 음성 분류: whisper-tiny 모델을 활용하여 뇌질환 특징에 따른 음성 분류
* STT: 음성 데이터를 텍스트화하여 통화 내용에 따른 뇌질환 분석
* RAG: 의학 지식을 참조하여 정확하며 쉬운 설명을 제공
* 요약: 다음 리포트 생성 시 참조할 요약(이전 상태와 변화 등)
* AI Agent: 자가 문단표와 분석 결과를 종합하여 최종 리포트 생성
* FastAPI: 백엔드 서버와 연결

## 파일 구조

```
AI_Server/
├── dataset/
│   ├── audio_cls_dataset.csv		# 뇌질환 음성
│   ├── normal_audio_dataset.csv	# 일반 노인 음성
│   └── rag_practicce.txt			# rag 실험용 음성
├── dummy/					# 테스트용 더미데이터
│   ├── dummydata1.json		
│   ├── dummydata2.json
│   ├── dummydata3.json
│   ├── dummydata4.json
│   └── dummydata5.json
├── model/				
│   ├── ai_agent_v3.py			# AI Agent 코드
│   └── whisper_cls.ipynb			# whisper tiny 학습 코드
├── .dockerignore
├── Dockerfile
├── fast_api.ipynb				# 실행 코드
├── main.py
└── requirements.txt
```

## Dependencies

```
langchain==0.3.0
langchain-community==0.3.0
langchain-core==0.3.0
langchain-openai
langchain-text-splitters
faiss-cpu
git+https://github.com/openai/whisper.git
fastapi
uvicorn
pyngrok
nest_asyncio
librosa
```
