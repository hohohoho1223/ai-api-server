from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

class AppModel:
  def __init__(self):
    load_dotenv() 
    self.model = init_chat_model("gpt-4o-mini", model_provider="openai") #요거랑
    system_template = "Translate the following from English into {language}"
    self.prompt_template = ChatPromptTemplate.from_messages(
      [("system", system_template), ("user", "{text}")]
    )#프롬프트에 메세지 역할이 여러가지가 있다.

  def get_response(self, message):
    return self.model.invoke([HumanMessage(message)])

  def get_prompt_response(self,language, message):
    prompt = self.prompt_template.invoke({"language": language, "text": message})
    return self.model.invoke(prompt)

def get_streaming_response(self, messages):
    return self.model.stream(messages) #비동기로함 