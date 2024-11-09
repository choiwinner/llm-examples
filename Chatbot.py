#실행 streamlit run Chatbot.py
#python version = 3.10.12
#langchain-google-genai =2.0.4
#streamlit =1.38.0
import streamlit as st
import os


#langchain, genai
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

with st.sidebar:
    gemini_api_key = st.text_input('Gemini_API_KEY를 입력하세요.', key="langchain_search_api_gemini", type="password")
    "[Get an Gemini API key](https://ai.google.dev/)"
    "[How to get Gemini API Key](https://luvris2.tistory.com/880)"

    if (gemini_api_key[0:2] != 'AI') or (len(gemini_api_key) != 39):
        st.warning('잘못된 key 입력', icon='⚠️')
    else:
        st.success('정상 key 입력', icon='👉')

    if process :=st.button("Process"):
        if (gemini_api_key[0:2] != 'AI') or (len(gemini_api_key) != 39):
            st.error("잘못된 key 입력입니다. 다시 입력해 주세요.")
            st.stop()

    if data_clear :=st.button("대화 클리어"):
        st.session_state['messages'] = [] #st.session_state[messages]를 초기화

    st.subheader('Gemini Model을 선택하세요.')
    selected_model = st.sidebar.selectbox('Choose Gemini Model', ['gemini-1.5-flash', 'gemini-1.5-flash-latest','gemini-1.5-pro', 'gemini-1.5-pro-latest'], key='selected_model')
    "gemini-1.5-flash : 분당 15회 요청 가능, 분당 100만 토큰 처리, 하루 1500회 요청 제한"
    "gemini-1.5-pro : 분당 2회 요청 가능, 분당 32,000 토큰 처리, 하루 50회 요청 제한"
    "[참고 자료 1](https://blog.naver.com/PostView.naver?blogId=itandtech&logNo=223624403146)"
    "[참고 자료 2](https://ai.google.dev/gemini-api/docs/models/gemini?hl=ko)"

st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by Gemini")

#0. gemini api key Setting
os.environ["GOOGLE_API_KEY"] = gemini_api_key

#1. st.session_state 초기화
if "messages" not in st.session_state:
    st.session_state['messages'] = [] #st.session_state에 messages가 없으면 빈 리스트로 초기화
    #2.'assistant' icon으로 write를 출력한다.

st.chat_message("assistant").write("안녕하세요. 무엇을 도와드릴까요?")

#2. 이전 대화 내용을 출력
# st.session_state['messages']가 있고 길이가 0 이상이면 실행 
if ("messages" in st.session_state) and (len(st.session_state['messages'])>0):
    for role, message in st.session_state['messages']:  #st.session_state['messages']는 tuple 형태로 저장되어 있음.
        st.chat_message(role).write(message)

#3. query를 입력받는다.
if query := st.chat_input("질문을 입력해주세요."): 
    if not gemini_api_key:
        st.warning("Gemini API Key를 입력해 주세요.")
        st.stop()

    #4.'user' icon으로 query를 출력한다.
    st.chat_message("user").write(query)
    #5. query를 session_state 'user'에 append 한다.
    st.session_state['messages'].append(('user',query))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chain = (
                ChatPromptTemplate.from_template(
                    """
                    당신은 질문에 답하는 Chatbot으로 질문에 대해 자세하게 답변하고 답변의 참고 문헌이나 URL 정보도 알려주세요.
                    [질문]
                    {question}
                    """
                    ) 
                    | ChatGoogleGenerativeAI(model=selected_model, temperature = 0) 
                    | StrOutputParser()
                    )
            # chain 호출
            response = chain.invoke({"question": query})
            st.write(response)
            st.session_state['messages'].append(('assistant',response))
            
# 참고 자료 
# 1. https://www.youtube.com/watch?v=VtS8yF2ItgI
# 2. https://wikidocs.net/book/14314