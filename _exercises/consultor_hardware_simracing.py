from langchain_ollama import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

llm = ChatOllama(model="gemma4:e2b", temperature=0.5)

### Bloco Basico - Chamada direta para testar o modelo e a resposta
# print("Consultor: Bem-vindo! Vou analisar seu hardware.\n")

# user_input = input("Digite sua pergunta: ")

# print("Processsando...\n")

# # response = llm.invoke(user_input)
# # print(response.content)
# # print("\n")
# # print(response.response_metadata)

# for chunk in llm.stream(user_input):
#     print(chunk.content, end="", flush=True)

# print("\n")
### FimBlocoBasico

### Bloco com Prompt Template e Streaming (LCEL Básico)
# user_input = input("Digite o(s) componente(s) de hardware que deseja analisar para simracing: ")

# print("Processsando...\n")

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "Você é um consultor de hardware especializado em simracing em PC Gamer. Responda de forma clara e objetiva."),
#     ("user", "Fale brevemente sobre os pros e contras de investir em {user_input} para simracing.")
# ])

# chain = prompt | llm | StrOutputParser()

# for chunk in chain.stream({"user_input": user_input}):
#     print(chunk, end="", flush=True)

# print("\n")
### FimBlocoLCEL

### Runnables, Roteamento e Lógica de Negócio
# chain_hardware = (
#     ChatPromptTemplate.from_template("Responda como técnico de PC: O usuário perguntou sobre {pergunta}") | llm | StrOutputParser()
# )

# chain_sim_racing = (
#     ChatPromptTemplate.from_template("Responda como piloto virtual: O usuário perguntou sobre {pergunta}") | llm | StrOutputParser()
# )

# def roteador(dicionario_entrada):
#     pergunta = dicionario_entrada["pergunta"].lower()

#     if "volante" in pergunta or "cockpit" in pergunta or "simulador" in pergunta:
#         print("[ROTA] -> Direcionando para Especialista em Sim Racing")
#         return chain_sim_racing
#     else:
#         print("[ROTA] -> Direcionando para Técnico de Hardware Generalista")
#         return chain_hardware
    
# chain_principal = RunnableLambda(roteador)

# input_user = input("Digite sua pergunta sobre hardware para simracing: ")

# for chunkk in chain_principal.stream({"pergunta": input_user}):
#     print(chunkk, end="", flush=True)

# print("\n")
### FimBlocoRunnables

