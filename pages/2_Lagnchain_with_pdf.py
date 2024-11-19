import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# PyPDFLoader시 리스트형태로 1Page씩 불러오고 이를 list.extend를 이용해서 리스트 하나씩(1Page씩 추가함)
from langchain.document_loaders import PyPDFLoader
import tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnableMap

def process_pdfs2(pdf_docs):
    with st.spinner("PDF 파일 처리 중..."):
        documents = []
        for uploaded_file in pdf_docs:
            # 원래 파일 이름 유지
            original_filename = uploaded_file.name

            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix=original_filename) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # PyPDFLoader를 사용하여 PDF 로드
            loader = PyPDFLoader(tmp_file_path)
            documents.extend(loader.load())

            # 임시 파일 삭제
            os.unlink(tmp_file_path)

            # 1000자로 청킹하기
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 50)

            #splitter를 이용한 문서 청킹
            data = text_splitter.split_documents(documents)

            #임베딩 하기
            #임베딩 모델
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            vectorstore = FAISS.from_documents(data, embeddings)

            st.success(f"총 {len(pdf_docs)}파일, {len(data)}개의 페이지를 성공적으로 로드했습니다.")
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    template = """You are an AI assistant specialized in answering questions based on the provided PDF document. 
    Use the following pieces of context to answer the human's question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    그리고 matadata를 참고해서 답변의 source도 알려주세요.

    {context}

    Human: {question}
    AI Assistant: """

    PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )
    
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=4) 
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"Human: {message.content}")
        else:
            st.write(f"AI: {message.content}")


def main():

    st.set_page_config(page_title="Lagnchain_with_pdf", page_icon=":books:")
    st.title("🦜🔗 Lagnchain_with_pdf")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state['chat_history'] = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "pdf_input" not in st.session_state:
        st.session_state.pdf_input = None 
    if "retriever_from_llM" not in st.session_state:
        st.session_state.retriever_from_llM = {}

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
            st.session_state['chat_history'] = []
            st.session_state.vectorstore = None
            st.session_state.pdf_input = None
            st.session_state.retriever_from_llM = {}
            
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
        pdf_docs = st.file_uploader("PDF 파일을 업로드하세요", type="pdf", accept_multiple_files=True)

        st.session_state.vectorstore = process_pdfs2(pdf_docs)
        
    
    retriever_from_llM = st.session_state.vectorstore.as_retriever(search_type="mmr",search_kwargs={'k':5, 'fetch_k': 10})

    template = """당신은 인공지능 ChatBOT으로 Question 내용에 대해서 대답합니다.
    대답은 Context에 있는 내용을 참조해서만 답변합니다.
    되도록이면 자세한 내용으로 대답하고 context의 있는 original source도 같이 보여주세요.
    Context: {context}
    Question: {question}
    """

    st.chat_message("assistant").write("안녕하세요. 무엇을 도와드릴까요?")

    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    chain = RunnableMap({
        "context": lambda x: retriever_from_llM.get_relevant_documents(x['question']),
        "question": lambda x: x['question']
    }) | prompt | llm

    #1. st.session_state 초기화
    if "messages" not in st.session_state:
        st.session_state['messages'] = [] #st.session_state에 messages가 없으면 빈 리스트로 초기화
        #2.'assistant' icon으로 write를 출력한다.

    #윈도우 크기 k를 지정하면 최근 k개의 대화만 기억하고 이전 대화는 삭제
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=4) 

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    #2. 이전 대화 내용을 출력
    # st.session_state['messages']가 있고 길이가 0 이상이면 실행 
    if ("messages" in st.session_state) and (len(st.session_state['messages'])>0):
        for role, message in st.session_state['messages']:  #st.session_state['messages']는 tuple 형태로 저장되어 있음.
            st.chat_message(role).write(message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # chain 호출
            response = chain.invoke({'question': query}).content
            st.write(response)
            st.session_state['messages'].append(('assistant',response))

























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
                response = st.session_state.conversation({'question': query})
                st.write(response['answer'])
                #st.session_state['chat_history'].append(('assistant',response['answer']))

        

##2. 이전 대화 내용을 출력
## st.session_state['messages']가 있고 길이가 0 이상이면 실행 
#if ("messages_pdf" in st.session_state) and (len(st.session_state['messages_pdf'])>0):
#    for role, message in st.session_state['messages_pdf']:  #st.session_state['messages']는 tuple 형태로 저장되어 #있음.
#        st.chat_message(role).write(message)
#
#
#from langchain_google_genai import ChatGoogleGenerativeAI
#
#llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
#
#from langchain.chains import ConversationalRetrievalChain
#conversation_chain = ConversationalRetrievalChain.from_llm(
#    llm=llm,
#    chain_type="stuff",
#    retriever=vectorstore.as_retriever(),
#)
#
##3. query를 입력받는다.
#if query := st.chat_input("질문을 입력해주세요."): 
#
#    #4.'user' icon으로 query를 출력한다.
#    st.chat_message("user").write(query)
#    #5. query를 session_state 'user'에 append 한다.
#    st.session_state['messages'].append(('user',query))
#
#    with st.chat_message("assistant"):
#        with st.spinner("Thinking..."):
#            # chain 호출
#            response = conversation_chain.run(question=query)
#            st.write(response)
#            st.session_state['messages'].append(('assistant',response))

if __name__ == '__main__':
    main()
    

    



