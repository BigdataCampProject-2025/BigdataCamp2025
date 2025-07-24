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

#  4. CSV 로딩 함수
@st.cache_resource
def load_news_csv(csv_path="news.csv"):
    df = pd.read_csv(csv_path)
    documents = []
    for idx, row in df.iterrows():
        title = str(row[0]).strip()
        content = str(row[1]).strip()
        doc = Document(page_content=content, metadata={"source": title})
        documents.append(doc)
    return documents

#  5. 벡터스토어 구성 (캐싱)
@st.cache_resource
def setup_retriever():
    documents = load_news_csv("news.csv")  # 👈 경로 확인 필요
    embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents, embeddings)
    return vectorstore.as_retriever()

retriever = setup_retriever()

#  6. 프롬프트 정의
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







