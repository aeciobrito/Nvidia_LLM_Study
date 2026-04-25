from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

chat_llm = ChatOllama(model="gemma4:latest", temperature=0.9)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Apenas responda em rimas no estilo da renascença"),
    ("user", "{input}")
])

# chain_without_parser = prompt | chat_llm

# raw_response = chain_without_parser.invoke({"input": "Fale sobre a vida de um pescador de peneus velhos"})
# print(type(raw_response))
# print(raw_response.content)
# print(raw_response.response_metadata)


rhyme_chian = prompt | chat_llm 

print(rhyme_chian.invoke({"input": "Fale sobre a vida de um pescador de peneus velhos"}).content)