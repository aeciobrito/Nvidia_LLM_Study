from langchain_ollama import ChatOllama
 

llm = ChatOllama(model="gemma4:e2b", temperature=0.3)

print("Consultor: Bem-vinddo! Vou analisar seu hardware.\n")

user_input = input("Digite sua pergunta: ")

print("Processsando...\n")

response = llm.invoke(user_input)
print(response.content)
print("\n")
print(response.response_metadata)

# for chunk in llm.stream(user_input):
#     print(chunk.content, end="", flush=True)

print("\n")