import gradio as gr
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_core.runnables import RunnableLambda

# --- 1. O MOLDE DA NOSSA MEMÓRIA (PYDANTIC) ---
class PerfilPiloto(BaseModel):
    orcamento: str = Field("desconhecido", description="Orçamento disponível. Ex: 5000 reais, baixo, alto. 'desconhecido' se não informado.")
    experiencia: str = Field("desconhecido", description="Nível de experiência em sim racing. Ex: iniciante, pro. 'desconhecido' se não informado.")
    setup_atual: str = Field("", description="O que o usuário já possui (ex: joga no teclado, tem um volante velho, etc).")

# --- 2. O EXTRATOR DE DADOS (SLOT FILLING COM ACUMULAÇÃO) ---
def criar_extrator():
    parser = PydanticOutputParser(pydantic_object=PerfilPiloto)
    
    prompt = ChatPromptTemplate.from_template(
        "Você é um assistente de extração de dados estrito.\n"
        "Sua tarefa é analisar a mensagem do usuário e atualizar o perfil dele.\n\n"
        "REGRAS CRÍTICAS:\n"
        "1. Nunca apague dados que já existem no PERFIL ATUAL.\n"
        "2. Se o usuário fornecer novas informações sobre o setup, ADICIONE essas novas informações ao que já existe no campo 'setup_atual' do PERFIL ATUAL.\n"
        "3. Apenas retorne o JSON final formatado, nada mais.\n\n"
        "{instrucoes}\n\n"
        "PERFIL ATUAL: {perfil_atual}\n"
        "MENSAGEM DO USUÁRIO: {input}\n\n"
        "NOVO PERFIL ACUMULADO (Apenas JSON):"
    )
        
    def limpar_json(texto):
        if '{' in texto: texto = texto[texto.find('{'):]
        if '}' in texto: texto = texto[:texto.rfind('}')+1]
        return texto

    return (RunnableAssign({'instrucoes': lambda x: parser.get_format_instructions()}) 
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

chain_interna = (
    RunnableAssign({'perfil_atual': extrator})
    | RunnableAssign({'diretriz_acao': RunnableLambda(decidir_acao)})
)

prompt_resposta = ChatPromptTemplate.from_messages([
    ("system", "Você é um consultor especializado em setups de Simulação e PC.\n"
               "MEMÓRIA DO CLIENTE: {perfil_atual}\n\n"
               "SUA DIRETRIZ AGORA É: {diretriz_acao}\n\n"
               "Não mencione que você tem diretrizes ou memória, apenas aja naturalmente."),
    ("user", "{input}")
])
chain_externa = prompt_resposta | ChatOllama(model="gemma4:latest", temperature=0.7) | StrOutputParser()

# =====================================================================
# --- 5. A PONTE COM O FRONTEND (Lógica do Gradio) ---
# =====================================================================

# Mantemos o estado global instanciado fora da função para atuar como um "Singleton" 
# de memória durante a sessão de uso.
estado_global = {'perfil_atual': PerfilPiloto(), 'input': ''}

def chat_streaming(message, history):
    global estado_global
    
    # 1. Recebe a nova mensagem da interface
    estado_global['input'] = message
    
    # 2. Roda a "Regra de Negócio" silenciosa (Atualiza JSON e decide a ação)
    estado_global = chain_interna.invoke(estado_global)
    
    # Print no console do Linux para você acompanhar os bastidores (Backoffice)
    print(f"\n[DEBUG DA MEMÓRIA]: {estado_global['perfil_atual'].model_dump()}")
    
    # 3. Gera a resposta para a interface gráfica em streaming
    buffer = ""
    for token in chain_externa.stream(estado_global):
        buffer += token
        yield buffer

# --- 6. INICIALIZAÇÃO DA INTERFACE GRÁFICA ---
if __name__ == "__main__":
    mensagem_inicial = [{"role": "assistant", "content": "Bem-vindo à Consultoria de Projetos Sim Racing! 🏎️\nPara começarmos, me conte o que você já tem de equipamento no seu PC."}]
    
    # Inicialização clássica, robusta e compatível!
    chatbot = gr.Chatbot(value=mensagem_inicial)
    demo = gr.ChatInterface(chat_streaming, chatbot=chatbot)
    
    demo.launch(share=True)