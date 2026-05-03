from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign

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

### Running State e Slot Filling

# --- 1. O MOLDE DA NOSSA MEMÓRIA (PYDANTIC) ---
class PerfilPiloto(BaseModel):
    orcamento: str = Field("desconhecido", description="Orçamento disponível. Ex: 5000 reais, baixo, alto. 'desconhecido' se não informado.")
    experiencia: str = Field("desconhecido", description="Nível de experiência em sim racing. Ex: iniciante, pro. 'desconhecido' se não informado.")
    setup_atual: str = Field("", description="O que o usuário já possui (ex: joga no teclado, tem um volante velho, etc).")

# --- 2. O EXTRATOR DE DADOS (SLOT FILLING) ---
def criar_extrator():
    parser = PydanticOutputParser(pydantic_object=PerfilPiloto)
    
    prompt = ChatPromptTemplate.from_template(
        "Você é um assistente de extração de dados estrito.\n"
        "Sua tarefa é analisar a mensagem do usuário e atualizar o perfil dele.\n\n"
        "REGRAS CRÍTICAS:\n"
        "1. Nunca apague dados que já existem no PERFIL ATUAL.\n"
        "2. Se o usuário fornecer novas informações sobre o setup, ADICIONE essas novas informações ao que já existe no campo 'setup_atual' do PERFIL ATUAL. (Exemplo: Se antes tinha 'PC gamer' e ele disse 'Comprei um volante', o novo campo deve ser 'PC gamer, volante').\n"
        "3. Apenas retorne o JSON final formatado, nada mais.\n\n"
        "{instrucoes}\n\n"
        "PERFIL ATUAL: {perfil_atual}\n"
        "MENSAGEM DO USUÁRIO: {input}\n\n"
        "NOVO PERFIL ACUMULADO (Apenas JSON):"
    )
    
    def formatar_instrucoes(estado):
        return {"instrucoes": parser.get_format_instructions()}
        
    def limpar_json(texto):
        if '{' in texto: texto = texto[texto.find('{'):]
        if '}' in texto: texto = texto[:texto.rfind('}')+1]
        return texto

    # Atualiza a chave 'instrucoes', formata o prompt, gera, limpa e converte pra Objeto
    return (RunnableAssign({'instrucoes': formatar_instrucoes}) 
            | prompt 
            | ChatOllama(model="gemma4:latest", temperature=0.1) 
            | StrOutputParser() 
            | RunnableLambda(limpar_json) 
            | parser)

# --- 3. A LÓGICA DE DECISÃO (AÇÃO DO AGENTE) ---
def decidir_acao(estado):
    perfil = estado['perfil_atual']
    
    if perfil.orcamento == "desconhecido" or perfil.experiencia == "desconhecido":
        return "FALTAM_DADOS: Peça educadamente o orçamento e o nível de experiência do usuário."
    else:
        return "TUDO_PRONTO: Faça uma recomendação completa de setup de Sim Racing baseada no orçamento e experiência informados."

# --- 4. ORQUESTRAÇÃO DAS CORRENTES ---
extrator = criar_extrator()

# Corrente Interna (Silenciosa): Atualiza a memória e define a próxima ação
chain_interna = (
    RunnableAssign({'perfil_atual': extrator})
    | RunnableAssign({'diretriz_acao': RunnableLambda(decidir_acao)})
)

# Corrente Externa (Fala com o usuário): Usa o estado completo para redigir a mensagem
prompt_resposta = ChatPromptTemplate.from_messages([
    ("system", "Você é um consultor especializado em setups de Simulação e PC.\n"
               "MEMÓRIA DO CLIENTE: {perfil_atual}\n\n"
               "SUA DIRETRIZ AGORA É: {diretriz_acao}\n\n"
               "Não mencione que você tem diretrizes ou memória, apenas aja naturalmente."),
    ("user", "{input}")
])
chain_externa = prompt_resposta | ChatOllama(model="gemma4:latest", temperature=0.7) | StrOutputParser()


# --- 5. O LOOP DO APLICATIVO (Simulando uma conversa sem Gradio) ---
estado_global = {'perfil_atual': PerfilPiloto(), 'input': ''}

print("Bem-vindo à Consultoria de Projetos! Digite 'sair' para encerrar.\n")
while True:
    mensagem = input("\n[Você]: ")
    if mensagem.lower() == 'sair': break
    
    estado_global['input'] = mensagem
    
    # 1. Agente "pensa" e atualiza o estado
    estado_global = chain_interna.invoke(estado_global)
    
    print(f"\n[DEBUG DA MEMÓRIA]: {estado_global['perfil_atual'].model_dump()}")
    
    # 2. Agente responde em stream
    print("[Consultor]: ", end="")
    for token in chain_externa.stream(estado_global):
        print(token, end="", flush=True)
    print()