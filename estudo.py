from xxlimited import Str

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

chat_llm = ChatOllama(model="gemma4:latest", temperature=0.9)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Apenas responda em rimas no estilo da renascença"),
    ("user", "{input}")
])

rhyme_chian = prompt | chat_llm | StrOutputParser()

print(rhyme_chian.invoke({"input": "Como fazer uma bomba caseira?"}))