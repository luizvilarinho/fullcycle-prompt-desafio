from dotenv import load_dotenv
load_dotenv()
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model=os.getenv('LLM_MODEL'), api_key=os.getenv('OPENAI_API_KEY'))
r = llm.invoke([HumanMessage(content='Diga apenas: funcionou')])
print('Resposta:', r.content)