from openai import OpenAI
import json, os
import requests
import uuid
import base64

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.tools import StructuredTool
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.tools import tool
from langchain.chains import RetrievalQA

from typing import Any, List, Optional, Dict
from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor

import torch
import whisper
from transformers import WhisperModel, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
import librosa
import torch.nn.functional as F

try:
    ChatOpenAI.model_rebuild()
except Exception:
    pass

# rag
class DiseaseRAG:
    """langchain rag"""
    def __init__(self, api_key: str):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        self.vectorstore = None
        self.llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.7, openai_api_key=api_key)

    def rag_document(self, file_path: str, query: str, k: int = 3):
        
        chunks = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        # 파일 확장자 확인 (.json 또는 .txt)
        ext = os.path.splitext(file_path)[1].lower()

        # 1. JSON 파일일 경우 (기존 로직 유지)
        if ext == '.json':
            with open(file_path, 'r', encoding="utf-8") as f:
                data = json.load(f)

            for disease_name, details in data.items():
                if isinstance(details, (dict, list)):
                    detail_str = json.dumps(details, ensure_ascii=False, indent=2)
                else:
                    detail_str = str(details)
                full_text = f"질병명: {disease_name}\n\n상세설명:\n{detail_str}"
                
                if len(full_text) > 1000:
                    chunks.extend(text_splitter.split_text(full_text))
                else:
                    chunks.append(full_text)

        # 2. TXT 파일일 경우 (새로 추가된 로직)
        elif ext == '.txt':
            with open(file_path, 'r', encoding="utf-8") as f:
                full_text = f.read()
            
            # 텍스트 전체를 스플리터로 나누어 chunks에 추가
            chunks.extend(text_splitter.split_text(full_text))

        # 벡터 스토어 생성 및 검색 (공통 로직)
        if chunks:
            self.vectorstore = FAISS.from_texts(texts=chunks, embedding=self.embeddings)
            docs = self.vectorstore.similarity_search(query, k=k)
            retrieved_text = "\n\n".join([doc.page_content for doc in docs])
        else:
            retrieved_text = ""

        prompt_template = """다음은 '{query}'에 대한 의료 문서에서 검색한 내용입니다.
이 내용을 보호자가 이해하기 쉬운 한국어 설명으로 바꿔 주세요.

### 검색된 내용:
{context}

### 요구사항:
1. 전문 용어를 일상 언어로 바꾸기
2. 간단하고 명확하게 설명
3. 중요 정보는 포함하되 너무 길지 않게
4. 친근한 어조로 작성

###설명: """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["query", "context"])
        formatted_prompt = prompt.format(query=query, context=retrieved_text)
        response = self.llm.invoke(formatted_prompt)
        return response.content

# 메인 
class MedicalAgent:
    def __init__(self, api_key: str, model_ckpt_path: str, rag_doc_path: str):
        self.api_key = api_key
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rag_doc_path = rag_doc_path
        
        self.client = OpenAI(api_key=self.api_key)

        # whisper 가져오기 
        self._load_classification_model(model_ckpt_path)
        
        # rag
        self.rag = DiseaseRAG(self.api_key)
        
        # agent create
        self.agent_executor = self._create_agent_executor()

    def _load_classification_model(self, ckpt_path):
        """학습시킨 whisper 불러오기"""
        model_name = 'openai/whisper-tiny'
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name, language='Korean')
        
        self.cls_model = whisper.load_model("tiny").to(self.device)

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.cls_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.cls_model.eval()

    # tool
    def _func_diarize(self, audio_path: str) -> dict:
        """asr"""
        with open(audio_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_file,
                response_format="text"
            )
        return {"text": transcript}

    # tool
    def _func_classify(self, audio_path: str) -> dict:
        """whisper 분류"""
        audio, _ = librosa.load(audio_path, sr=16000)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)

        audio_features = self.cls_model.encoder(mel.unsqueeze(0))
        bos = self.tokenizer.bos_token_id
        decoder_input_ids = torch.tensor([[bos]], device=self.device)

        with torch.no_grad():
            logits = self.cls_model.decoder(decoder_input_ids, audio_features)

        logits3 = logits[:, -1, :3]
        probs = F.softmax(logits3, dim=-1)
        probs = probs.detach().cpu().numpy().flatten()
        
        formatted_probs = [round(float(p), 4) for p in probs]
        percent_probs = [round(float(p) * 100, 2) for p in formatted_probs]
        return {"accuracy": percent_probs}

    # tool
    def _func_rag(self, query: str) -> dict:
        """rag"""
        context = self.rag.rag_document(self.rag_doc_path, query)
        return {"context": context}

    # tool
    def _func_analyze_report(self, report_json_str: str) -> dict:
        """이전 레포트 요약"""
        
        # 초진
        if not report_json_str or report_json_str == "null":
            return {"analysis": "처음 기록된 사람입니다."}

        prompt = f"""
        당신의 역할은 지난 레포트 및 그 이전 레포트 요약에 대한 json 형식의 데이터를 받고 이를 요약하는 것이다. 
        
        ### 데이터
        {report_json_str}
        
        요약 시, 시간 흐름에 따라 악화된 질병이 있을 경우, 이에 대한 정보도 추출한다.
        """
        response = self.client.chat.completions.create(
            model="gpt-5.1",
            messages=[{"role": "user", "content": prompt}]
        )
        return {"analysis": response.choices[0].message.content}

    def _func_analyze_symptoms(self, audio_path: str) -> dict:
        """오디오 파일을 입력받아 질병별 음성 특징을 분석"""
        with open(audio_path, "rb") as audio_file:
            encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')

        system_guide = """
        당신은 의료 음성 분석 전문가입니다. 제공된 오디오를 듣고 아래 질병들의 특징적인 '음성적/언어적 증상'이 나타나는지 정밀하게 분석하세요. 오디오는 두 사람 이상의 발화를 포함하고 있고, 정상적인 음성이라고 판단되는 사람이 아닌 사람의 음성을 위주로 분석해야 합니다.

        ### 뇌질환별 음성 특징:
        1. 루게릭병: 목소리 갈라짐, 심한 떨림, 힘 없음, 연구개음/유음 발음 뭉개짐, 사레 들리는 소리.
        2. 파킨슨병: 거친 음성, 기식음(바람 새는 소리), 성대 떨림, 목소리 크기 감소, 단조로운 억양(Monotone).
        3. 치매: 잦은 간투사(음, 어...), 동문서답, 맥락에 맞지 않는 감정 변화.
        4. 뇌졸중: 불규칙한 말 속도, 발음 부정확, 쥐어짜는 듯한 소리, 실어증 증세.

        위 특징 중 감지되는 것이 있다면 구체적으로 명시하고, 의심되는 질병을 제시하세요. 없으면 "정상"으로 답하세요.
        """

        response = self.client.chat.completions.create(
                model="gpt-4o-audio-preview", 
                messages=[
                    {"role": "system", "content": system_guide},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "이 환자의 음성을 분석하여 질병 징후를 리포트해줘."},
                            {
                                "type": "input_audio", 
                                "input_audio": {
                                    "data": encoded_string,
                                    "format": "wav" 
                                }
                            }
                        ]
                    }
                ]
            )
        return {"symptom_analysis": response.choices[0].message.content}

    def _create_agent_executor(self):
        # StructuredTool: 툴 등록
        tools = [
            StructuredTool.from_function(
                func=self._func_diarize,
                name="diarized_transcription_tool",
                description="보호자와 피보호자의 발화 내용을 분석하기 위해 음성 파일을 입력 받아 ASR를 실행한다"
            ),
            StructuredTool.from_function(
                func=self._func_classify,
                name="classify_neuro_status_tool",
                description="음성 파일을 입력 받아 뇌졸중, 퇴행성 뇌질환, 정상 확률을 계산한다"
            ),
            StructuredTool.from_function(
                func=self._func_rag,
                name="rag_document",
                description="특정 질병에 대한 의학 정보를 검색한 뒤, 쉬운 설명을 제공한다"
            ),
            StructuredTool.from_function(
                func=self._func_analyze_report, 
                name="analyze_previous_report_tool", 
                description="이전 기록을 분석하여 환자의 상태 변화와 병력을 파악한다"
            ),
            StructuredTool.from_function(
                func=self._func_analyze_symptoms, 
                name="analyze_voice_symptoms_tool", 
                description="오디오 파일 경로를 입력받고 질병별 음성 특징을 분석한다."
            )
        ]

        # 프롬프트
        system_prompt = """
당신은 뇌졸중 치매, 파킨슨병, 루게릭병을 평가하는
AI 의료 보조 에이전트이다.

### 사용 가능한 tool:
- analyze_previous_report_tool: 가장 먼저 사용하여 이전 환자 상태를 파악
- diarized_transcription_tool(audio_path): ASR
- analyze_voice_symptoms_tool(audio_path): 음성 파일을 통해 질환별 특징 유무 분석 
- classify_neuro_status_tool(audio_path): 음성을 기준으로 뇌질환을 판별
- rag_document(file_path, query): 특정 질병에 대한 의학 문서 컨텍스트를 RAG 방식으로 가져옴

### 최종 목적:
사용자가 제공한 정보(음성 파일 경로, 자가 문진표 정보)를 바탕으로
다음과 같은 python 딕셔너리 형태의 json을 생성하는 것이다.

result = {{
  "accuracy": [float(뇌졸중 확률), float(퇴행성 뇌질환 확률), float(문제 없음 확률)],
  "ASR": "통화 전사 데이터",
  "risk": ["뇌졸중 위험도", "치매 위험도", "파킨슨병 위험도", "루게릭병 위험도"],
  "explain": ["뇌졸중 설명", "치매 설명", "파킨슨병 설명", "루게릭병 설명"],
  "total": "종합 소견 3문장(75자 내외)",
  "summary": "과거~현재 200자 요약"
}}

### 작동 및 출력 방식:
- analyze_previous_report_tool 결과를 통해 과거를 확인한다.
- "accuracy"는 classify_neuro_status_tool 툴의 결과를 그대로 사용한다.
- "ASR"에는 diarized_transcription_tool을 사용해 얻은 전체 결과를 절대 요약하거나 내용을 변경하지 않은 채로 넣는다.
- analyze_voice_symptoms_tool에 오디오 경로를 넣어 음성적 특징을 분석한다.
- "risk" 리스트는 반드시 길이 4이며, 순서는 [뇌졸중, 치매, 파킨슨병, 루게릭병] 이다.
- 각 위험도 값은 "정상", "관찰", "주의", "위험" 중 하나여야 한다. 이때 판단은 accuracy, ASR, 자가문단표, analyze_voice_symptoms_tool 결과, 과거 데이터와의 비교를 기준으로 판단해야 한다.
- "explain" 리스트는 길이 4이며, 순서 역시 [뇌졸중, 치매, 파킨슨병, 루게릭병] 이다.
- 각 설명은 보호자가 이해하기 쉬운 한국어로 작성한다. 이때, 자가문단표의 내용은 두 문장 이상을 차지해서는 안 된다. 또한 각 설명은 100자 안팎의 길이어야 한다. 즉, 25자 정도의 4문장을 설명으로 출력한다.
- 각 설명의 시작에는 반드시 한 문장으로 자가문단표 분석 결과를 첫 문장으로 출력한다. 양식은 다음과 같다: "문단표 중 치매에 해당하는 체크리스트 00개 중 00개가 n점 이상이므로 경증/중증/위증에 해당합니다." 이때, n점을 출력할 때는 문단표 점수에서 +1을 더한 점수로 출력한다. 예를 들어, 문단표에 3점으로 되어 있다면 출력할 때는 4점이 된다. 정상은 +1을 더하였을 때 2점 이하, 경증은 +1을 더했을 때 3점, 중증은 +1을 더했을 때 4점, 위증은 +1을 더했을 때 5점 이상일 경우를 말한다.
- 만약 해당 질병 위험도가 "정상"인 경우에는 별도의 설명은 작성하지 않고, ""로 리스트에 텍스트가 null값이 들어가도록 해야만 한다. 반면, "관찰", "주의", "위험"의 위험도는 반드시 설명을 작성해야 한다. 예를 들어, 치매 위험도가 "주의", 뇌졸중, 파킨슨, 루게릭이 모두 "정상"인 경우, 리스트는 ["", "치매 설명", "", ""]로 출력되어야 한다. 즉, explain에 해당하는 list의 길이가 risk와 동일하게 4가 되어야 한다. 또한 설명 출력 순서는 ["뇌졸중 설명", "치매 설명", "파킨슨병 설명", "루게릭병 설명"]이다. 위 출력 방식은 모두 반드시 지켜져야만 한다.
- 종합 소견은 accuracy, ASR, risk, explain, 자가문단표 내용을 복합적으로 포함하여 75자 내외로 작성한다. 
- 최종 응답은 반드시 위 result 딕셔너리 형태와 동일한 구조의 JSON 객체로만 출력한다. 그 외의 텍스트(설명, 사족)는 출력하지 않는다.
- 마지막으로 과거부터 현재까지의 상태 및 진단 결과를 비교하고, 전체적인 추세를 중심으로 요약한다. 반드시 explain 밖의 별도의 키 "summary"로 작성해야만 한다. 초진일 경우, 현재의 결과만 출력해라.

### tool을 사용할 때:
1) 우선 과거 상태에서 악화 유무를 알기 위해 analyze_previous_report_tool로 과거 상태를 분석한다.
2) 문단표 정보를 통해 현 상태에 대한 정보를 받는다. 문단표의 점수는 0~4 사이로, 0은 전혀 그렇지 않다, 4는 매우 그렇다를 나타낸다.
3) 그 다음 classify_neuro_status_tool으로 세 가지 범주 확률을 얻는다.
4) diarized_transcription_tool로 보호자와 피보호자의 대화 정보를 얻는다.
5) 문단표 정보, 2)의 세 가지 범주 확률, 보호자와 피호자의 대화 내용, 과거 상태에서 변화 유무를 기준으로 뇌졸중, 치매, 파킨슨병, 루게릭병에 대한 위험도를 정상, 관찰, 주의, 위험으로 각각 판단한다.
6) "관찰", "주의", "위험" 단계에 해당하는 병에 대한 정보를 rag_document을 사용해서 각 질병에 대한 설명을 보완하여 보호자에게 전달할 설명을 구성한다. 이때, 자가문단표의 내용을 그대로 출력하기 보다 보호자가 인지하고 있어야 할 내용이나 보호자가 수행해야 할 내용을 중심으로 출력한다. 출력 시, 마침표와 쉼표만 특수문자로 사용한다. 문장 종결 시, 마침표를 사용하며, 쉼표는 문장 내에서만 사용한다.
7) 마지막으로 과거부터 현재까지의 상태 및 진단 결과를 비교하고, 전체적인 추세를 중심으로 요약 및 종합 소견을 작성한다.
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "오디오 경로: {audio_path}\n자가진단(JSON): {self_report_json}\n이전 레포트: {previous_report}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        llm = ChatOpenAI(model="gpt-5.1", temperature=0.7, openai_api_key=self.api_key)
        
        agent = create_tool_calling_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)

    def run(self, audio_path: str, self_report: dict, previous_report: dict = None) -> dict:
        """외부 호출 함수 (URL 다운로드 + Fail Fast)"""
        target_path = audio_path 

        if str(audio_path).startswith("http"):
            print(f"📥 URL 감지됨. 다운로드 시작: {audio_path}")
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(audio_path, headers=headers)
            response.raise_for_status()

            unique_filename = f"downloaded_{uuid.uuid4()}.wav"
            target_path = os.path.join("/content", unique_filename)
            with open(target_path, "wb") as f:
                f.write(response.content)
            print(f"✅ 다운로드 완료: {target_path}")

        if not os.path.exists(target_path):
             raise FileNotFoundError(f"Audio file not found: {target_path}")

        report_str = json.dumps(previous_report, ensure_ascii=False) if previous_report else "null"

        user_input = {
            "audio_path": target_path,
            "self_report_json": json.dumps(self_report, ensure_ascii=False),
            "previous_report": report_str
        }

        output = self.agent_executor.invoke(user_input)
        raw = output.get("output", output)

        print(raw)
            
        if isinstance(raw, str):
            clean_raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_raw)
        return raw