import streamlit as st
from PIL import Image
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
        st.session_state['messages_img'] = [] #st.session_state[messages]를 초기화

st.title("📝 Image Q&A with Gemini")

#0. gemini api key Setting
if not gemini_api_key:
    st.warning("Gemini API Key를 입력해 주세요.")
    st.stop()

genai.configure(api_key=gemini_api_key)

#1. st.session_state 초기화
if "messages_img" not in st.session_state:
    st.session_state['messages_img'] = [] #st.session_state에 messages가 없으면 빈 리스트로 초기화
    #2.'assistant' icon으로 write를 출력한다.

# 파일 업로더 위젯
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

# 파일이 업로드되면 처리
if uploaded_file is not None:

    # 파일을 바이너리 데이터로 읽기
    img_bytes = uploaded_file.read()

    # PIL로 이미지를 열고 스트림릿 앱에 표시
    #image = Image.open(uploaded_file).resize((300,300))
    image = Image.open(uploaded_file)
    #st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.image(image, caption="Uploaded Image.")
    st.write("Image uploaded successfully!")
else:
    st.write("Please upload a JPG or PNG file.")

# 파일을 지정한 경로에 저장 (임시 파일 경로)
with open("temp_image.jpg", "wb") as f:
    f.write(img_bytes)
    
# genai.upload_file 사용하여 llm에 파일 업로드
uploaded_file_info = genai.upload_file(path="temp_image.jpg", display_name="uploaded_image")

## Choose a Gemini API model.
model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

#1. st.session_state 초기화
if "messages_img" not in st.session_state:
    st.session_state['messages_img'] = [] #st.session_state에 messages가 없으면 빈 리스트로 초기화
    #2.'assistant' icon으로 write를 출력한다.

st.chat_message("assistant").write("안녕하세요. 무엇을 도와드릴까요?")

#2. 이전 대화 내용을 출력
# st.session_state['messages']가 있고 길이가 0 이상이면 실행 
if ("messages_img" in st.session_state) and (len(st.session_state['messages_img'])>0):
    for role, message in st.session_state['messages_img']:  #st.session_state['messages']는 tuple 형태로 저장되어 있음.
        st.chat_message(role).write(message)

#3. query를 입력받는다.
if query := st.chat_input("질문을 입력해주세요."): 

    #4.'user' icon으로 query를 출력한다.
    st.chat_message("user").write(query)
    #5. query를 session_state 'user'에 append 한다.
    st.session_state['messages_img'].append(('user',query))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # llm에 사진에 대해서 물어 본다.
            response = model.generate_content([uploaded_file_info, query])
            st.write(response.text)
            st.session_state['messages_img'].append(('assistant',response.text))


