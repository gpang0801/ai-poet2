__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os
import streamlit as st
import tempfile

load_dotenv()

st.title("ChatPDF")
st.write("---")

upload_file = st.file_uploader("PDF 파일을 업로드 해주세요", type=["pdf"])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader(temp_filepath)
    pages = loader.load()  # ✅ 여기서는 load() 추천
    return pages

# ✅ 업로드 안 했으면 여기서 종료 (pages 미정의 방지)
if upload_file is None:
    st.info("먼저 PDF를 업로드 해주세요.")
    st.stop()

pages = pdf_to_document(upload_file)

st.header("질문을 입력해주세요")
question = st.text_input("질문 입력")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False
)

texts = text_splitter.split_documents(pages)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
db = Chroma.from_documents(texts, embeddings)

llm = ChatOpenAI(temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(),
    llm=llm
)

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": RunnableLambda(lambda q: retriever_from_llm.invoke(q)) | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

if st.button("질문하기"):
    with st.spinner("답변 생성 중..."):
        if not question.strip():
            st.warning("질문을 입력해주세요.")
        else:
            result = rag_chain.invoke(question)
            st.write(result)
