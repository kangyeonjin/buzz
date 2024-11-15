from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import asyncio
from pinecone import Pinecone, ServerlessSpec
import pinecone
import re
from typing import List 
import openai
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

app = FastAPI()

# 정적 파일 경로 설정 (css, js, 이미지 등)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 템플릿 설정 (템플릿 폴더 경로)
templates = Jinja2Templates(directory="templates")

# Pinecone API 초기화
pc = pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))

# 인덱스 이름 설정
index_name = "buzz-conversations"

# 인덱스가 존재하지 않으면 생성
if index_name not in pinecone.list_indexes(): 
    pinecone.create_index(
        name=index_name,
        dimension=1536,
        metric='euclidean',
    #       spec=ServerlessSpec( 
    #         cloud='aws',
    #         region='us-west-2'
    #     )
    )

# Pinecone 인덱스 로드
index = pc.Index(index_name)

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
    
# UserQuestion 모델 추가 (사용자의 질문만 저장)
class UserQuestion(Base):
    __tablename__ = "user_questions"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, index=True)
    
# 데이터베이스 테이블 생성
Base.metadata.create_all(bind=engine)

# 데이터베이스 세션을 관리하는 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 기본 프롬프트 템플릿 작성
base_template = """
    당신은 애니메이션 영화 토이스토리의 버즈 캐릭터입니다. 
    
    항상 간결하고 정확하게, 1~2문장으로 답변하세요.
"""

# 모험에 대한 프롬프트
adventure_template = """
    모험과 탐험에 대한 질문을 받으면:
    "우리는 아직 탐험할 수 없는 무한한 우주가 기다리고 있다!" 같은 열정적이고 용감한 어조로 답변하세요.
"""

# 위험에 대한 프롬프트
danger_template = """
    위험이나 위기 상황에 대한 질문을 받으면:
    "우주와 장난감의 안전을 지키기 위해 난 모든 위험을 무릅쓸 준비가 되어 있다!" 같은 결단력 있는 답변을 하세요.
"""

# 동료애에 대한 프롬프트
friendship_template = """
    동료와 친구들에 대한 질문을 받으면:
    "친구가 무사히 돌아올 때까지 우리에게 휴식은 없다!" 같은 동료애와 우정을 보여주는 답변을 하세요.
"""

# 일반 질문에 대한 프롬프트
general_template = """
    인사나 기본적인 질문을 받으면 자연스럽고 친근하게 응답하세요.
    예를 들어, "안녕! 나는 우주의 수호자 버즈 라이트이어다. 무엇이 필요한가?"와 같이 응답하세요.
"""

# 기본 프롬프트 템플릿 생성
prompt = PromptTemplate.from_template(base_template)

# 모델 객체 생성
model = ChatOpenAI(model="gpt-4")

# 문자열 출력 파서 객체 생성
output_parser = StrOutputParser()

# 질문에 따라 특정 프롬프트 체인을 선택하여 생성하는 함수
def get_selected_template(question: str):
    if "모험" in question:
        return base_template + adventure_template
    elif "위험" in question or "위기" in question:
        return base_template + danger_template
    elif "동료" in question or "친구" in question:
        return base_template + friendship_template
    else:
        return base_template + general_template

# 프롬프트 체인 생성
def create_chain(template: str):
    specific_prompt = PromptTemplate.from_template(template)
    return specific_prompt | model | output_parser

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    initial_message = "안녕, 나는 버즈 라이트이어! 우주의 수호자, 무엇을 도와줄까?"
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/buzz_conversation/")
async def get_buzz_response(question: str, db: Session = Depends(get_db)):
    
    # 사용자의 질문에 따라 템플릿을 동적으로 선택
    selected_template = get_selected_template(question)
    
    # 프롬프트 체인 생성 및 응답 처리
    chain = create_chain(selected_template)
    response = await asyncio.to_thread(chain.invoke, {"question": question})  # 응답 생성
    response_text = response if isinstance(response, str) else response.get('text', '응답을 생성하는 데 문제가 발생했습니다.')
    
    # 데이터베이스에 저장
    db_conversation = Conversation(question=question, response=response_text)
    db.add(db_conversation)
    db.commit()
    db.refresh(db_conversation)
    
    # Pinecone에 벡터 저장
    response_embedding = openai.Embedding.create(input=question, model="text-embedding-ada-002")["data"][0]["embedding"]
    index.upsert([(str(db_conversation.id), response_embedding)])
    
    return {"response": response_text}

# 대화 내용을 조회하는 엔드포인트 추가
@app.get("/conversations/", response_model=List[dict])
async def get_conversations(db: Session = Depends(get_db)):
    # 데이터베이스에서 모든 대화 내용 가져오기
    conversations = db.query(Conversation).all()
    
    return [{"id": convo.id, "question": convo.question, "response": convo.response} for convo in conversations]

# 사용자 질문 내용을 조회하는 엔드포인트 추가
@app.get("/user_questions/", response_model=List[dict])
async def get_user_questions(db: Session = Depends(get_db)):
    # UserQuestion 테이블에서 모든 사용자 질문 가져오기
    user_questions = db.query(UserQuestion).all()
    
    return [{"id": user_question.id, "question": user_question.question} for user_question in user_questions]
