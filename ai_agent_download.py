import json
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

def init_clients(api_key: str):
    client = OpenAI(api_key=api_key)

    llm_json = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0,
        openai_api_key=api_key,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    return client, llm_json


# tool
def build_tools(client):

    @tool
    def diarized_transcription_tool(audio_path: str) -> dict:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe-diarize",
                file=audio_file,
                response_format="text",
            )
        return {"asr_text": transcript}

    @tool
    def classify_neuro_status_tool(audio_path: str) -> dict:
        '''
        일단 임시
        '''
        probs = audio_classifier(audio_path)
        stroke_prob, degenerative_prob, normal_prob = probs

        return {
            "accuracy": [float(stroke_prob), float(degenerative_prob), float(normal_prob)],
            "reason": "음성 패턴 기반 신경학적 분석 결과 (placeholder)"
        }

    @tool
    def retrieve_disease_info_tool(query: str) -> dict:
        context = (
            f"[RAG PLACEHOLDER] (query: {query}) 에 대한 의학 문서 요약 컨텍스트입니다. "
            "실제 서비스에서는 벡터DB 기반 RAG 결과가 여기에 들어갑니다."
        )
        return {"context": context}

    return [
        diarized_transcription_tool,
        classify_neuro_status_tool,
        retrieve_disease_info_tool,
    ]

def build_prompt() -> ChatPromptTemplate:

    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
너는 뇌졸중 및 퇴행성 뇌질환(치매, 파킨슨병, 루게릭병)을 평가하는
AI 의료 보조 에이전트이다.

너는 아래와 같은 tools를 사용할 수 있다:
- diarized_transcription_tool(audio_path): 음성 파일을 받아 ASR 텍스트를 얻는다.
- classify_neuro_status_tool(audio_path): 음성을 기준으로 뇌질환을 판별하여
  [뇌졸중, 퇴행성 뇌질환, 문제 없음] 확률 3개를 반환한다.
- retrieve_disease_info_tool(query): 특정 질병에 대한 의학 문서 컨텍스트를
  RAG 방식으로 가져오는 placeholder이다.

최종적으로 너의 목적은,
사용자가 제공한 정보(음성 파일 경로, 자가 문진표 정보)를 바탕으로
다음과 같은 Python 딕셔너리 형태의 JSON을 생성하는 것이다.

result = {{
  "accuracy": [float(뇌졸중 확률), float(퇴행성 뇌질환 확률), float(문제 없음 확률)],
  "ASR": "통화 전사 데이터",
  "risk": ["뇌졸중 위험도", "치매 위험도", "파킨슨병 위험도", "루게릭병 위험도"],
  "explain": ["뇌졸중 설명", "치매 설명", "파킨슨병 설명", "루게릭병 설명"]
}}

제약 조건:
- "accuracy"는 classify_neuro_status_tool 툴의 결과를 그대로 사용한다.
- "ASR"에는 diarized_transcription_tool을 사용해 얻은 전체 텍스트를 넣는다.
- "risk" 리스트는 반드시 길이 4이며, 순서는
  [뇌졸중, 치매, 파킨슨병, 루게릭병] 이다.
- 각 위험도 값은 "정상", "관찰", "주의", "위험" 중 하나여야 한다.
  이때 판단은 accuracy, ASR, 자가 문진표를 기준으로 판단해야 한다.
- "explain" 리스트는 길이 4이며, 순서 역시
  [뇌졸중, 치매, 파킨슨병, 루게릭병] 이다.
- 각 설명은 보호자가 이해하기 쉬운 한국어로 작성한다.
- 만약 해당 질병 위험도가 "정상"인 경우에도 설명은 작성하되,
  "현재로서는 특이 소견이 없어 보입니다" 등 안심할 수 있는 안내를 제공한다.
- 최종 응답은 반드시 위 result 딕셔너리 형태와 동일한 구조의 JSON 객체로만 출력한다.
  그 외의 텍스트(설명, 사족)는 출력하지 않는다.

tool을 사용할 때:
1) 자가 문진표 정보를 참고하여 환자의 전반적인 상태를 이해한다.
2) diarized_transcription_tool을 사용해 ASR 텍스트를 얻는다.
3) classify_neuro_status_tool으로 세 가지 범주 확률을 얻는다.
4) 필요하다면 retrieve_disease_info_tool을 사용해서 각 질병에 대한 설명을 보완한다.
5) 자가 문진표(self_report) 정보를 고려하여 질병별 위험도를 결정하고,
   보호자에게 전달할 설명을 구성한다.
"""
        ),
        (
            "user",
            """
다음 정보를 바탕으로 result를 생성해줘.

- audio_path: {audio_path}
- self_report(JSON): {self_report_json}

위 audio_path를 사용해 tools를 호출해서 ASR과 확률을 계산하고,
자가 문진표 정보를 반영하여 최종 result를 만들어.
"""
        ),
    ])

def build_agent(api_key: str) -> AgentExecutor:
    """
    api_key를 받아 OpenAI client와 tools, prompt를 생성하고
    이를 이용해 LangChain tool-calling agent를 구성한 뒤 AgentExecutor를 반환합니다.
    """
    client, _ = init_clients(api_key)
    tools = build_tools(client)
    agent_prompt = build_prompt()

    agent_llm = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0,
        openai_api_key=api_key,
    )

    agent = create_tool_calling_agent(agent_llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    return agent_executor

def run_agent_with_function_calling(api_key: str, audio_path: str, self_report: dict) -> dict:

    agent_executor = build_agent(api_key)

    user_input = {
        "audio_path": audio_path,
        "self_report_json": json.dumps(self_report, ensure_ascii=False),
    }

    output = agent_executor.invoke(user_input)
    raw = output.get("output", output)

    if isinstance(raw, str):
        result = json.loads(raw)
    else:
        result = raw

    # 간단한 유효성 검사
    required_keys = ["accuracy", "ASR", "risk", "explain"]
    for key in required_keys:
        if key not in result:
            raise ValueError(f"결과에 '{key}' 키가 없습니다. 실제 결과: {result}")

    return result