<<<<<<< HEAD
from tkinter import Image
import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
import tiktoken
import yfinance as yf
import pandas as pd
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from summa.summarizer import summarize

from langgraph.graph import StateGraph, END, START
from typing import Literal, TypedDict

# -------------------- 1. 환경 변수 설정 --------------------
load_dotenv()
=======

import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate

# ⬛️ 1. 환경 변수 설정
load_dotenv()

>>>>>>> 0481fc31e4eb6c70463597a9a707896e8dbcd360
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION")

<<<<<<< HEAD
# -------------------- 2. Streamlit 설정 --------------------
st.set_page_config(page_title="주식PLUS", page_icon="📈")
st.title("📈 TOPPIC")
=======
# ⬛️ 2. Streamlit 앱 시작
st.set_page_config(page_title="RAG 기반 주식 챗봇", page_icon="📈")
st.title("📈 이번 주 주목할 주식 챗봇")
>>>>>>> 0481fc31e4eb6c70463597a9a707896e8dbcd360

# ⬛️ 3. 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

<<<<<<< HEAD
# -------------------- 3. 문서 압축 함수 --------------------
def compress_document(doc: Document, ratio: float = 0.3) -> Document:
    original_content = doc.page_content
    title = doc.metadata.get("source", "제목 없음")
    compressed_content = summarize(original_content, ratio=ratio)
    if not compressed_content or len(compressed_content.strip()) < 50:
        compressed_content = original_content
    return Document(page_content=compressed_content, metadata={"source": title})


# -------------------- 실시간 주식 정보 --------------------

def calculate_indicators(df, ticker):
    close = df[('Close', ticker)]
    volume = df[('Volume', ticker)] if ('Volume', ticker) in df.columns else pd.Series(dtype=float)

    df[('Indicators', 'SMA_5')] = close.rolling(window=5).mean()
    df[('Indicators', 'SMA_20')] = close.rolling(window=20).mean()

    # RSI 계산
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    df[('Indicators', 'RSI')] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df[('Indicators', 'MACD')] = macd
    df[('Indicators', 'Signal')] = signal

    # 거래량 평균과 비교
    if not volume.empty and volume.notna().any():
        volume_ma5 = volume.rolling(window=5).mean()
        volume_ratio = volume / (volume_ma5 + 1e-6)
        df[('Indicators', 'Volume_MA5')] = volume_ma5
        df[('Indicators', 'Volume_ratio')] = volume_ratio
    else:
        df[('Indicators', 'Volume')] = 0
        df[('Indicators', 'Volume_MA5')] = 1
        df[('Indicators', 'Volume_ratio')] = 1

    # 전일 대비 수익률
    df[('Indicators', 'Daily_Return')] = close.pct_change()

    return df

def check_surge_possibility(df, ticker):
    output = ""
    latest = df.iloc[-1]

    conds = {
        "골든크로스": latest[('Indicators', 'SMA_5')] > latest[('Indicators', 'SMA_20')],
        "RSI 양호": 50 < latest[('Indicators', 'RSI')] < 70,
        "MACD 상승": latest[('Indicators', 'MACD')] > latest[('Indicators', 'Signal')],
        "거래량 급증": latest[('Indicators', 'Volume_ratio')] > 1.5,
        "전일 대비 +5%": latest[('Indicators', 'Daily_Return')] > 0.05
    }

    score = sum(conds.values())

    output += "\n📊 기술지표 기반 급등 가능성 평가:\n"
    for k, v in conds.items():
        output += f" - {k}: {'✅' if v else '❌'}\n"

    if score >= 4:
        output += "\n🚀 급등 가능성 높음 (기술지표 기준)\n\n"
    elif score >= 2:
        output += "\n⚠️ 약한 상승 신호 존재\n\n"
    else:
        output += "\n🔎 급등 가능성 낮음 (기술적 지표 기준)\n\n"

    return output, conds

def analyze_ticker(ticker, period='7d', interval='15m'):
    output = f"🔍 [{ticker}] 데이터 분석 중...\n"

    df = yf.download(ticker, period=period, interval=interval)
    if df.empty:
        output += "❌ 데이터를 불러올 수 없습니다.\n"
        return output, None

    output += f"{df.head().to_string()}\n"

    df = calculate_indicators(df, ticker)
    surge_output, conds = check_surge_possibility(df, ticker)
    output += surge_output
    output += f"\n📈 최근 5일간 데이터:\n{df.tail(5).to_string()}\n"
    return output
# -------------------- 4. 뉴스 CSV 로드 및 문서 생성 --------------------
=======
# ⬛️ 4. CSV 로딩 함수
>>>>>>> 0481fc31e4eb6c70463597a9a707896e8dbcd360
@st.cache_resource
def load_news_csv(csv_path="news.csv"):
    df = pd.read_csv(csv_path)
    documents = []
<<<<<<< HEAD
    for _, row in df.iterrows():        
        title = str(row[0]).strip()
        content = str(row[1]).strip()
        doc = Document(page_content=content, metadata={"source": title})
        compressed_doc = compress_document(doc, ratio=0.2)
        documents.append(compressed_doc)
    return documents

# -------------------- 5. 벡터스토어 설정 --------------------
@st.cache_resource
def setup_retriever():
    documents = load_news_csv("yfinance_articles.csv")
=======
    for idx, row in df.iterrows():
        title = str(row[0]).strip()
        content = str(row[1]).strip()
        doc = Document(page_content=content, metadata={"source": title})
        documents.append(doc)
    return documents

# ⬛️ 5. 벡터스토어 구성 (캐싱)
@st.cache_resource
def setup_retriever():
    documents = load_news_csv("news.csv")  # 👈 경로 확인 필요
>>>>>>> 0481fc31e4eb6c70463597a9a707896e8dbcd360
    embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents, embeddings)
    return vectorstore.as_retriever()

retriever = setup_retriever()

<<<<<<< HEAD
# -------------------- 6. 프롬프트 템플릿 --------------------
prompt_recommendation = PromptTemplate.from_template("""
당신은 주식 전문가입니다. 아래는 주식 관련 뉴스 기사들입니다.

이번 주 핫한 주식 관련 질문에 답변해 주세요. 아래 4가지 정보를 포함해 주세요:

- 종목명
- 설명
- 이유
- 출처(참고했던 정확한 뉴스 제목)

📚 뉴스 기사:
{context}

❓ 질문:
{question}
""")

prompt_stock_info = PromptTemplate.from_template("""
당신은 주식 전문가입니다. 아래는 주식 관련 뉴스 기사와 기술적 분석 리포트입니다.
기술적 분석 리포트의 내용을 바탕으로, 리포트의 평가도 추가해주세요
특정 종목 관련 질문에 대해 아래 정보를 포함해 답변해 주세요:

- 종목명
- 정보
- 최신주가
- 비전, 전망, 근거
- 기술적 분석 리포트
- 출처(정확한 뉴스 제목)

📚 뉴스 기사:
{context}

📉 기술적 분석 리포트:
{data}

❓ 질문:
{question}
""")

# -------------------- 7. 문서 포맷터 --------------------
def format_docs(docs):
    return "\n\n".join(
        f"-{i+1}-\n{doc.page_content}\n[출처: {doc.metadata.get('source', '알 수 없음')}]"
        for i, doc in enumerate(docs)
    )

# -------------------- 8. LangGraph 상태 정의 --------------------
class RAGState(TypedDict):
    question: str
    context: str
    prompt: str
    data: str
    answer: str

# -------------------- 9. LangGraph 노드 정의 --------------------
def retrieve_node(state: RAGState) -> dict:
    docs = retriever.invoke(state["question"])
    return {"context": format_docs(docs)}

def classify_node(state: RAGState) -> dict:
    llm = AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=api_base,
        api_version=api_version,
        deployment_name="gpt-4o-mini",
        temperature=0.0
    )
    classification_prompt = PromptTemplate.from_template("""
아래 질문을 읽고 어떤 유형의 질문인지 파악하시오:
- 종합추천: 이번주 화제,주목받는 주식의 정보나 주식 추천이나 투자 전략에 대한 질문 (이번주, 주목받는,핫한,등의 키워드면 반드시 종합추천으로 분류)
        종합추천 예시:이번주 주목받은 주식, 어떤 주식이 가장 핫한가요?,투자할 만한 주식은?
- 종목정보: 특정 주식이나 종목에 대한 정보 요청,비전 ,전망,근거 등 
        종목정보 예시: 삼성전자 주가, 애플의 비전과 전망은?, 현대차의 최근 주가는?
답변은 반드시 아래의 두 가지 중 하나로만 해주세요:
- 종합추천
- 종목정보

질문:
{question}
""")
    prompt_text = classification_prompt.format(question=state["question"])
    response = llm.invoke(prompt_text)
    print(f"분류 결과: {response.content.strip()}")
    if "종합추천" in response.content.strip():
        print("종합추천으로 분류됨")
        return {"type": "종합추천"}
    else:
        print("종목정보로 분류됨")
        return {"type": "종목정보"}
    
def prompt_recommendation_node(state: RAGState) -> dict:
    prompt_text = prompt_recommendation.format(
        context=state["context"],
        question=state["question"]
    )
    return {"prompt": prompt_text}

# def prompt_stock_info_node(state: RAGState) -> dict:
#     prompt_text = prompt_stock_info.format(
#         context=state["context"],
#         question=state["question"]
#     )
#     return {"prompt": prompt_text}

def prompt_stock_info_node(state: RAGState) -> dict:
    # 1. 종목명 추출
    ticker_name = extract_ticker_name(state["question"])

    # 2. 종목 분석 리포트 생성
    try:
        report_text = analyze_ticker(ticker_name)
    except Exception as e:
        report_text = f"{ticker_name}에 대한 기술 분석 정보를 가져오는 데 실패했습니다.\n오류: {e}"
    print(f"종목 분석 리포트: {report_text}")
    state["data"] = report_text

    # 3. 프롬프트 생성
    prompt_text = prompt_stock_info.format(
        context=state["context"],
        data=state["data"],
        question=state["question"]
    )
    return {"prompt": prompt_text}


def llm_node(state: RAGState) -> dict:
    llm = AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=api_base,
        api_version=api_version,
        deployment_name="gpt-4o-mini",
        temperature=0.5
    )
    response = llm.invoke(state["prompt"])
    return {"answer": response.content}



def extract_ticker_name(question: str) -> str:
    llm = AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=api_base,
        api_version=api_version,
        deployment_name="gpt-4o-mini",
        temperature=0.2
    )
    prompt = PromptTemplate.from_template("""
질문에서 주식 종목명을 한글이든 영어든 정확하게 한 가지만 추출해 주세요.
예시: " 구글 주가 어때?" → "GOOGL", "애플의 비전은?" → "AAPL" , "마이크로소프트 전망?" → "MSFT"
반드시 종목명만 출력하세요. (불필요한 설명 금지)

질문:
{question}
""")
    response = llm.invoke(prompt.format(question=question))
    return response.content.strip()

# -------------------- 10. LangGraph 구성 --------------------
workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("classify", classify_node)
workflow.add_node("prompt_recommendation", prompt_recommendation_node)
workflow.add_node("prompt_stock_info", prompt_stock_info_node)
workflow.add_node("llm", llm_node)

workflow.add_edge(START,"retrieve")
workflow.add_edge("retrieve", "classify")
workflow.add_conditional_edges(
    "classify",
    lambda state: state["type"],
    {
        "종합추천": "prompt_recommendation",
        "종목정보": "prompt_stock_info"
    }
)
workflow.add_edge("prompt_recommendation", "llm")
workflow.add_edge("prompt_stock_info", "llm")
workflow.add_edge("llm", END)

rag_graph = workflow.compile()

# -------------------- 11. 그래프 이미지 저장 --------------------
graph = rag_graph.get_graph().draw_mermaid_png()
with open("rag_workflow.png", "wb") as f:
    f.write(graph)

# -------------------- 12. Streamlit 인터페이스 --------------------
=======
# ⬛️ 6. 프롬프트 정의
prompt = PromptTemplate.from_template(
    """
다음은 여러 뉴스 기사입니다. 각 문서에는 해당 기사 본문과 함께 [출처: 제목]이 명시되어 있습니다.

이 기사들의 내용만을 사용해서 질문에 대답해줘
각 종목에 대해 다음 정보를 포함해줘:
1. 종목명 (가능하면 종목코드)
2. 설명 (무슨 일이 있었는지)
3. 이유 (왜 주목해야 하는지)
4. 출처: 참고한 뉴스 기사 제목

중복된 출처는 종목끼리 공유해도 좋아. 정확히 어떤 문서에서 근거를 찾았는지 연결해서 말해줘.

뉴스 기사 모음:
{context}

질문:
{question}
"""
)

# ⬛️ 7. 문서 포맷터
def format_docs(docs):
    return "\n\n".join(
        f"-{i+1}-\n{doc.page_content}\n[출처: {doc.metadata.get('source', '알 수 없음')}]" 
        for i, doc in enumerate(docs)
    )

# ⬛️ 8. LLM과 RAG 체인 구성
llm = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=api_base,
    api_version=api_version,
    deployment_name="gpt-4o-mini",  # Azure에서 설정한 이름
    temperature=0.5
)

rag_chain = (
    RunnableParallel({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }) | prompt | llm | StrOutputParser()
)

# ⬛️ 9. 기존 메시지 표시
>>>>>>> 0481fc31e4eb6c70463597a9a707896e8dbcd360
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ⬛️ 10. 사용자 입력 받기
user_input = st.chat_input("무엇이든 물어보세요!")
if user_input:
<<<<<<< HEAD
=======
    # 입력 저장 및 출력
>>>>>>> 0481fc31e4eb6c70463597a9a707896e8dbcd360
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("답변 생성 중..."):
        try:
<<<<<<< HEAD
            result = rag_graph.invoke({"question": user_input})
            answer = result["answer"]
=======
            answer = rag_chain.invoke(user_input)
>>>>>>> 0481fc31e4eb6c70463597a9a707896e8dbcd360
            st.chat_message("assistant").write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"오류 발생: {e}")
<<<<<<< HEAD
=======




>>>>>>> 0481fc31e4eb6c70463597a9a707896e8dbcd360
