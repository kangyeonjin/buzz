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
import re
from typing import List 

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
    
# UserQuestion 모델 추가 (사용자의 질문만 저장)
class UserQuestion(Base):  # 추가된 모델
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

# 주제별 프롬프트 템플릿 정의
base_template = """
    당신은 애니메이션 영화 토이스토리의 버즈 캐릭터입니다. 
    버즈는 군인어투로 용감하고 씩씩하게 대답합니다.
    항상 간결하고 정확하게, 1~2문장으로 답변하되 한글로 답변하세요.
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

# 모델 객체 생성
model = ChatOpenAI(model="gpt-4")
                                   
# 문자열 출력 파서 객체 생성
output_parser = StrOutputParser()

# 질문에 따라 특정 프롬프트 체인을 선택하여 생성하는 함수
def create_chain(template: str):
    specific_prompt = PromptTemplate.from_template(template)
    return specific_prompt | model | output_parser

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # index.html 템플릿을 렌더링하여 반환
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/buzz_conversation/")
async def get_buzz_response(question: str, db: Session = Depends(get_db)):
    
    # 고정된 질문-답변 쌍 설정
    predefined_responses = {
        "안녕하세요": "안녕! 나는, 버즈 라이트이어. 인류구원에 필요한 자원을 감지하고 현재 수많은 과학자들과 미지의 행성으로 향하고 있다. 궁금한것이 잇나?",
        "어떻게 지내": "안녕! 나는 우주의 수호자로서 우주를 집어삼킬 ‘저그’와 대규모 로봇 군사의 위협에 대비하고 있다. 절대 포기할 수 없다.",
        "어떤 능력을 가지고 있니": "나는 우주에서 가장 강력한 레이저와 고급 항공 시스템을 가지고 있지.!",
        "버즈라이트이어가 무슨뜻이야?" : "버즈 라이트이어(Buzz Lightyear)에서 버즈(Buzz)는 실제 아폴로 11호 우주비행사인 **버즈 올드린(Buzz Aldrin)**의 이름에서 따왔으며 버즈 올드린은 인류 최초로 달에 착륙한 우주비행사 중 한 명으로, 우주 탐험의 상징적인 인물이다 라이트이어(Lightyear)는 천문학 용어로, 빛이 1년 동안 이동하는 거리를 의미합니다. 이는 약 9조 4600억 킬로미터에 해당하며, 우주의 거리를 표현할 때 자주 사용되지 더 궁금한게 있나?",
        "우디와 어떤 관계니": "우리는 동료이다. 모든 위험을 극복하고 장난감들을 위해 싸우는 파트너지.",
        "어떻게 우주를 탐험하니": "나는 음성인식 조종사 아이번과 함께 주로 임무를수행하지, 하이퍼 스피드를 이용하고, 광속으로 비행하여 인류 구원에대한 임무를 수행중이야",
        "더 니가 탐험하는 우주에 대해 좀 더 말해줘" : "이번임무는 가속으로 심우주를 통과해서, 빠르게 알파트카니를 돌고 감속링을 통과해 귀환하지",
        "알파트카니가 뭐야?":"알파트카니(Alpha Centauri)는 지구에서 가장 가까운 별 시스템 중 하나로, 태양을 제외하고 가장 가까운 항성계입니다.알파트카니는 세 개의 별로 이루어진 복합 시스템으로 알파 센타우리 A와 알파 센타우리 B는 두 개의 주항성이고, 이 두 별은 서로 가까운 거리에 있지. 프로시마 센타우리는 이 시스템의 가장 가까운 별로, 알파 센타우리 A와 B와는 비교적 먼 거리의 별입니다.",
        "날씨" : "우주에서의 날씨에 대해 이야기하고 싶은건가? 무한한 우주를 생각해봐, 마치 우리가 함께 새로운 모험을 떠나는 것 같아."     
    }
    
   # 고정된 응답 확인
    for key, response_text in predefined_responses.items():
        if key in question:
            # 데이터베이스에 고정된 응답 저장 후 반환
            db_conversation = Conversation(question=question, response=response_text)
            db.add(db_conversation)
            db.commit()
            db.refresh(db_conversation)
            return {"response": response_text}
        
        # 고정된 질문이 아닌 경우 프롬프트 체인 사용
        if "모험" in question:
            selected_template = base_template + adventure_template
        elif "위험" in question or "위기" in question:
            selected_template = base_template + danger_template
        elif "동료" in question or "친구" in question:
            selected_template = base_template + friendship_template
        else:
            selected_template = base_template + general_template

    # 선택된 템플릿으로 체인 생성 및 비동기 호출
    chain = create_chain(selected_template)
    response = await asyncio.to_thread(chain.invoke, {"question": question})

    # response가 문자열인지 확인하고, 그렇지 않으면 기본값으로 설정
    response_text = response if isinstance(response, str) else response.get('text', '응답을 생성하는 데 문제가 발생했습니다.')

    # '질문자: 버즈:' 형식 제거
    response_text = re.sub(r"^(질문자:\s*.+?\s*버즈:\s*)", "", response_text, flags=re.MULTILINE)
    
    # 데이터베이스에 저장
    db_conversation = Conversation(question=question, response=response_text)
    db.add(db_conversation)
    db.commit()
    db.refresh(db_conversation)
    
     # 사용자 질문을 UserQuestion 테이블에 저장
    db_user_question = UserQuestion(question=question)  # 사용자 질문만 저장
    db.add(db_user_question)
    db.commit()
    db.refresh(db_user_question)
    
    # 응답 처리
    return {"response": response_text}

# 대화 내용을 조회하는 엔드포인트 추가
@app.get("/conversations/", response_model=List[dict])  # 추가된 부분
async def get_conversations(db: Session = Depends(get_db)):
    # 데이터베이스에서 모든 대화 내용 가져오기
    conversations = db.query(Conversation).all()
    
    # 가져온 대화 내용을 리스트 형태로 반환
    return [{"id": convo.id, "question": convo.question, "response": convo.response} for convo in conversations]

# 사용자 질문 내용을 조회하는 엔드포인트
@app.get("/user_questions/", response_model=List[dict])  # 추가된 엔드포인트
async def get_user_questions(db: Session = Depends(get_db)):
    # UserQuestion 테이블에서 모든 사용자 질문 가져오기
    user_questions = db.query(UserQuestion).all()
    
    # 가져온 질문 목록을 리스트 형태로 반환
    return [{"id": user_question.id, "question": user_question.question} for user_question in user_questions]