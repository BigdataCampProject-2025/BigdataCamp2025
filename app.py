import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
import tiktoken
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from summa.summarizer import summarize

# 1. 환경 변수 설정
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION")

#  2. Streamlit 앱 시작
st.set_page_config(page_title="RAG 기반 주식 챗봇", page_icon="📈")
st.title("📈 이번 주 주목할 주식 챗봇")

#  3. 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []


def compress_document(doc: Document, ratio: float = 0.3) -> Document:
    """
    뉴스 본문만 요약하고 제목은 그대로 유지합니다.
    :param doc: 요약할 Document 객체
    :param ratio: 요약 비율 (0.2는 약 20%로 압축)
    :return: 요약된 Document 객체
    """
    original_content = doc.page_content
    title = doc.metadata.get("source", "제목 없음")

    # 본문 요약
    compressed_content = summarize(original_content, ratio=ratio)

    # 예외 처리: 요약이 너무 짧거나 실패한 경우 원문 사용
    if not compressed_content or len(compressed_content.strip()) < 50:
        compressed_content = original_content

    return Document(page_content=compressed_content, metadata={"source": title})


#  4. CSV 로딩 함수
@st.cache_resource
def load_news_csv(csv_path="news.csv"):
    df = pd.read_csv(csv_path)
    documents = []

    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0

    for idx, row in df.iterrows():
        title = str(row[0]).strip()  # 제목
        content = str(row[1]).strip()  # 본문

        # 제목은 그대로, 본문만 Document로
        doc = Document(page_content=content, metadata={"source": title})
        compressed_doc = compress_document(doc, ratio=0.2)

        # 토큰 수 계산 (압축 후)
        token_count = len(encoding.encode(compressed_doc.page_content))
        total_tokens += token_count

        documents.append(compressed_doc)

    print(f"전체 요약 후 토큰 수 합계: {total_tokens}")
    return documents



#  5. 벡터스토어 구성 (캐싱)
@st.cache_resource
def setup_retriever():
    documents = load_news_csv("yfinance_articles.csv")
    print(f"Number of chunks: {len(documents)}")
    embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents, embeddings)
    return vectorstore.as_retriever()

retriever = setup_retriever()




#  6. 프롬프트 정의
prompt = PromptTemplate.from_template(
    """
당신은 주식 전문가입니다. 아래는 주식 관련 뉴스 기사들입니다. 각 문서에는 해당 기사 본문과 함께 [출처: 기사 제목]이 명시되어 있습니다.

이 기사들의 내용을 참고해서 사용자 질문에 정확히 응답해주세요.

---

🟩 질문 유형에 따라 아래의 출력 형식을 반드시 따르세요:

1. **질문이 "이번 주 핫한 주식", "주목할만한 종목", "지금 관심가질 주식"과 관련된 경우**  
→ 각 종목에 대해 아래 4가지 정보를 포함해 주세요:

    - **종목명**: 명확한 회사명이나 주식명 등 식별 가능한 이름 (특정 기업명이 없는 경우 해당 종목은 추천하지 마세요)
    - **설명**: 무슨 일이 있었는지, 어떤 변화나 이벤트가 있었는지
    - **이유**: 왜 주목해야 하는지, 뉴스 내용을 종합한 판단
    - **출처**: 참고한 뉴스 기사 제목들 (여러 출처가 있으면 모두 나열)

2. **질문이 특정 종목에 대한 것인 경우 (예: "삼성전자 어때?", "카카오 전망 알려줘")**  
→ 해당 종목에 대해 아래 4가지 정보를 포함해 주세요:

    - **종목명**: 질문한 회사 또는 주식명 (명확하게 표시)
    - **정보**: 어떤 사업을 하고 있으며, 현재 어떤 상황인지
    - **최종 판단**: 질문에 대한 종합적 답변, 전망이나 해석
    - **출처**: 참고한 뉴스 기사 제목들 (정확히 어떤 기사에서 나온 내용인지 명시)

---

🔸 동일한 종목에 대한 출처가 여러 개일 경우, **한 종목 아래에 출처를 모두 나열**하세요.  
🔸 요청 수보다 검색 결과가 적을 경우, **찾은 만큼만 제공**해주세요.

---

📚 뉴스 기사 모음:
{context}

❓ 질문:
{question}

"""
)

#  7. 문서 포맷터
def format_docs(docs):
    return "\n\n".join(
        f"-{i+1}-\n{doc.page_content}\n[출처: {doc.metadata.get('source', '알 수 없음')}]" 
        for i, doc in enumerate(docs)
    )

#  8. LLM과 RAG 체인 구성
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

#  9. 기존 메시지 표시
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

#  10. 사용자 입력 받기
user_input = st.chat_input("무엇이든 물어보세요!")

if user_input:
    # 입력 저장 및 출력
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("답변 생성 중..."):
        try:
            answer = rag_chain.invoke(user_input)
            st.chat_message("assistant").write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"오류 발생: {e}")







