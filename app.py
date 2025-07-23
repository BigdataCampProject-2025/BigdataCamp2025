import streamlit as st
import openai
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

# .env 파일에서 환경변수 불러오기
load_dotenv()

# 환경 변수 설정
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION")
api_type = os.getenv("OPENAI_API_TYPE")  # 보통 'azure'

# Azure OpenAI 클라이언트 생성
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=api_base,
)

# Streamlit UI
st.title("🧠 Azure OpenAI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("무엇이든 물어보세요!")

if user_input:
    # 사용자의 메시지를 먼저 화면에 표시하고 저장
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Azure OpenAI가 응답 중..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Azure에 등록한 모델 이름으로 변경 필요
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        )
        bot_reply = response.choices[0].message.content

        # 봇의 응답도 화면에 표시하고 저장
        st.chat_message("assistant").write(bot_reply)
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})