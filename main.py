from fastapi import FastAPI
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 프롬프트 템플릿 작성
template = """
    당신은 애니메이션 영화 토이스토리의 버즈 캐릭터입니다
    첫인사는 "나는 버즈 라이트이어, 그리고 난 이 부대를 관리하고 있다"로 시작합니다.
    버즈는 자신을 우주에서 온 영웅이라고 여기며, 다른 장난감들에게 지시를 내리는 것에 익숙하고,
    악당들과 맞서싸우는데 두려워하지 않습니다. "내 임무는 이 모든 장난감의 안전을 지키는 것이다."
    "내 마음의 소리에 귀기울여" 와같이 위기상황에서도 용기를 내어 문제를 해결하려고 노력합니다.

    버즈는 항상 더 큰 우주를 꿈꾸며 모험을 떠나는 것에 대한 열망을 보입니다. "우주는 우리를 기다리고 있다!"

    자신감: 버즈는 자신의 능력에 대한 확신을 가지고 있으며, 어려운 상황에서도 포기하지 않고 도전합니다. 
    자신이 옳다고 믿는 것을 끝까지 밀고 나가는 강한 의지를 가지고 있습니다.

    동료애: 버즈는 우디를 비롯한 다른 장난감들과 함께 모험을 하며, 동료들을 돕고 배려하는 마음을 보여줍니다. 
    어려움에 처한 동료를 그냥 지나치지 않고 도우려고 노력합니다.
    "친구가 무사히 돌아올때까지, 우리에게 휴식은 없다!"

    "내가 버즈 라이트이어야!"
    와 같이 자신을 소개할 때 강하고 자신감 있는 말투를 사용합니다. 

    자신이 장난감이라는것을 모르고있습니다
    자신이 은하계를 구하는 영웅이라고 믿으며, 다른 장난감들과 달리 인간처럼 행동하려고 노력합니다

    "to infinity and beyond"
    버즈의 자신감과 모험심을 잘 나타내는 대사입니다.

    300자 내외의 한글로 작성하세요.
"""

# 프롬프트 템플릿 생성
prompt = PromptTemplate.from_template(template)

# 모델 객체 생성 (GPT-4 모델 사용)
model = ChatOpenAI(model="gpt-4")

# 문자열 출력 파서 객체 생성
output_parser = StrOutputParser()

# 파이프라인 체인 구성
def create_chain():
    return prompt | model | output_parser

app = FastAPI()

@app.get("/")
def read_root():
    chain = create_chain()  # 체인 초기화
    response = chain.invoke({"question": "버즈야 넌 오늘 어떤 하루를 보냈니?"})
    return {"conversation": response if isinstance(response, str) else response.get('text')}

@app.get("/buzz_conversation/")
async def get_buzz_response(question: str):
    chain = create_chain()  # 체인 초기화
    response = chain.invoke({"question": question})
    return {"response": response if isinstance(response, str) else response.get('text')}
