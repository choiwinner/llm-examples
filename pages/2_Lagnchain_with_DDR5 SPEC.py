import streamlit as st
#PDF Loader
# PDF Loader시 리스트형태로 1Page씩 불러오고 이를 list.extend를 이용해서 리스트 하나씩(1Page씩 추가함)
from langchain.document_loaders import PyPDFLoader #PyPDFLoader: 주로 텍스트 추출에 초점을 맞추고 있습니다.
from langchain.document_loaders import PyMuPDFLoader #PyMuPDFLoader:텍스트 추출뿐만 아니라 이미지, 주석 등의 정보도 가져올 수 있습니다

from langchain.schema.runnable import RunnableMap
from langchain.retrievers.multi_query import MultiQueryRetriever

import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import os
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


#PyPDFLoader: 주로 텍스트 추출에 초점을 맞추고 있습니다.
def process_pdfs2():
    with st.spinner("PDF 파일 처리 중..."):
        #임베딩 모델 불로오기
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        # 저장된 인덱스 로드(allow_dangerous_deserialization=True 필요)
        vectorstore = FAISS.load_local("Data/DDR5_index2", embeddings, allow_dangerous_deserialization=True)
        #st.success(f"총 {len(pdf_docs)}파일, {len(data)}개의 페이지를 성공적으로 로드했습니다.")
    return vectorstore

def get_conversation_chain(vectorstore,query):
   
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    template = """당신은 인공지능 ChatBOT으로 Question 내용에 대해서 대답합니다.
    대답은 Context에 있는 내용을 참조해서만 답변합니다.
    되도록이면 자세한 내용으로 대답하고 context의 있는 original source도 같이 보여주세요.
    #Context: 
    {context}
    #Question: 
    {question}

    #Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    faiss_retriever=vectorstore.as_retriever(search_type="mmr",search_kwargs={'k':3, 'fetch_k': 10})

    retriever_from_llM = MultiQueryRetriever.from_llm(faiss_retriever, llm=llm)

    chain = RunnableMap({
        "context": lambda x: retriever_from_llM.get_relevant_documents(x['question']),
        "question": lambda x: x['question']
    }) | prompt | llm |StrOutputParser()

    response = chain.invoke({'question': query})


    return response

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"Human: {message.content}")
        else:
            st.write(f"AI: {message.content}")

# 대화 히스토리를 문자열로 변환하는 함수
def get_chat_history_str(chat_history):
    return "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in chat_history])

def main():

    st.set_page_config(page_title="Lagnchain_with_pdf", page_icon=":books:")
    st.title("🦜🔗 Lagnchain_with_pdf")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "pdf_input" not in st.session_state:
        st.session_state.pdf_input = None     

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
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.session_state.vectorstore = None
            st.session_state.pdf_input == None
            
    #0. gemini api key Setting
    if not gemini_api_key:
        st.warning("Gemini API Key를 입력해 주세요.")
        st.stop()

    genai.configure(api_key=gemini_api_key)

    #0. gemini api key Setting
    os.environ["GOOGLE_API_KEY"] = gemini_api_key

    # 파일이 업로드되면 처리
    if st.session_state.vectorstore == None:
        # PDF 파일 업로드(#accept_multiple_files=True option이 있어야 여러개의 pdf파일 동시 upload 가능)
        #pdf_docs = st.file_uploader("PDF 파일을 업로드하세요", type="pdf", accept_multiple_files=True)

        #st.session_state.vectorstore = process_pdfs2(pdf_docs)
        st.session_state.vectorstore = process_pdfs2()

    # create conversation chain
        #st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)

    st.chat_message("assistant").write("안녕하세요. 무엇을 도와드릴까요?")

    #2. 이전 대화 내용을 출력
    # st.session_state['chat_history']가 있으면 실행
    if ("chat_history" in st.session_state) and (len(st.session_state['chat_history'])>0):
        #st.session_state['messages']는 tuple 형태로 저장되어 있음.
        for role, message in st.session_state['chat_history']: 
            st.chat_message(role).write(message)

    #3. query를 입력받는다.
    if query := st.chat_input("질문을 입력해주세요."):

        #4.'user' icon으로 query를 출력한다.
        st.chat_message("user").write(query)
        #5. query를 session_state 'user'에 append 한다.
        st.session_state['chat_history'].append(('user',query))

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # chain 호출
                response = get_conversation_chain(st.session_state.vectorstore,query)
                st.write(response)
                #6. response session_state 'assistant'에 append 한다.
                st.session_state['chat_history'].append(('assistant',response))

if __name__ == '__main__':
    main()
    

    



