from dotenv import load_dotenv

load_dotenv() #환경설정 불러오기ㅋㅋ(.env에서)

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai") # openai 사용

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("나는 섹시해!"),
]

print(model.invoke(messages))

# for token in model.stream(messages):
#     print(token.content, end="|")

from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}" #템플릿 만드는 코드

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")] # 동적으로 넣는거
)

prompt = prompt_template.invoke({"language": "Korean", "text": "I wanna go GYM now"})#프론프트 만든거
response = model.invoke(prompt)
print(response.content)

### 여기까지 해놓으면 로컬에서만 돌리지, 밖에선 못돌림