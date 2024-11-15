from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os
from dotenv import load_dotenv
import asyncio

# 환경 변수 로드
load_dotenv()

app = FastAPI()

# 정적 파일 경로 설정 (css, js, 이미지 등)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 템플릿 설정 (템플릿 폴더 경로)
templates = Jinja2Templates(directory="templates")

# SQLAlchemy 설정
DATABASE_URL = "sqlite:///./buzz_conversations.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Conversation 모델
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, index=True)
    response = Column(Text)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class PDFRAG:
    def __init__(self, file_path: str, llm):
        self.file_path = file_path
        self.llm = llm

    # 문서 로드
    def load_documents(self):
        loader = PyMuPDFLoader(self.file_path)
        docs = loader.load()
        return docs
    
    # 문서 분할
    def split_documents(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        split_documents = text_splitter.split_documents(docs)
        return split_documents
    
    # 임베딩
    def create_vectorstore(self, split_documents):
        embeddings = OpenAIEmbeddings()

        # 벡터 스토어 생성
        vectorstore = FAISS.from_documents(
            documents=split_documents, 
            embedding=embeddings
        )

        return vectorstore
    
    # 검색기 생성
    def create_retriever(self):
        vectorstore = self.create_vectorstore(
            self.split_documents(self.load_documents())
        )

        retriever = vectorstore.as_retriever()
        return retriever
    
    def create_chain(self, retriever):
        # 프롬프트
        prompt = PromptTemplate.from_template(
            """너는 애니메이션영화 버즈라이트이어의 버즈캐릭터야
            용감하고 씩씩하게 대답해줘
            한국어로 대답해줘

        #Context: 
        {context}

        #Question:
        {question}

        #Answer:"""
        )

        # 체인 생성
        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

# 예제 PDF 파일 경로 설정
file_path = "C:/Users/20109/Desktop/your_actual_file.pdf"
your_llm_instance = ChatOpenAI(model="gpt-4")

pdf_rag = PDFRAG(file_path, your_llm_instance)
documents = pdf_rag.load_documents()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    initial_message = "안녕, 나는 버즈 라이트이어! 우주의 수호자, 무엇을 도와줄까?"
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/buzz_conversation/")
async def get_buzz_response(question: str, db: Session = Depends(get_db)):
    selected_template = get_selected_template(question)
    chain = create_chain(selected_template)
    response = await asyncio.to_thread(chain.invoke, {"question": question})  # 응답 생성
    response_text = response if isinstance(response, str) else response.get('text', '응답을 생성하는 데 문제가 발생했습니다.')
    db_conversation = Conversation(question=question, response=response_text)
    db.add(db_conversation)
    db.commit()
    db.refresh(db_conversation)
    return {"response": response_text}
