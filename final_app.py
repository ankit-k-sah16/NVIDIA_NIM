from openai import OpenAI
import streamlit as st
import os

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()


#Lodaing the NVIDIA Api key
os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')
llm=ChatNVIDIA(model="meta/llama-3_1-70b-instruct")


def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=NVIDIAEmbeddings()
        st.session_state.loader=PyPDFLoader("sample.pdf")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecusiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50)
        st.session_state.fianl_document=st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_document,st.session_state.embeddings)


st.title("NVIDIA LLM Integration with Langchain/ NVIDIA NIM DEMO")
CHAT_PROMPT=ChatPromptTemplate.from_template("ANSWER THE QUESTION BASED ON THE BELOW DOCUMENTS: {context} QUESTION: {user_input}")


prompt1=st.text_input("Enter your question here based on the documents")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("Vector embedding of the documents is completed. FIASS vector store DB is ready for retrieval based on the documents.")

if prompt1:
    document_chain=create_stuff_documents_chain(llm,CHAT_PROMPT)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input': prompt1})
    print("Response time:", time.process_time()-start)
    st.write(response['answer'])


    ## with streamlit expander
    with st.exapnder("Document Similarity Search"):
        # Find the relevant chunks
        for i ,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("----------------------------------------------")




