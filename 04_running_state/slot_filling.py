import gradio as gr
from pydantic import BaseModel, Field
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign

# 1. Configuração do Modelo Local (Usando o Qwen ou Gemma que você tem)
# Dica: Para extração de JSON (Knowledge Base), temperature=0 é ideal para evitar alucinações
instruct_llm = ChatOllama(model="gemma4:latest", temperature=0.1)
chat_llm = ChatOllama(model="gemma4:latest", temperature=0.7)

# 2. O Banco de Dados "Mock" (Simulado) fornecido pela NVIDIA
def get_flight_info(d: dict) -> str:
    """Simula uma consulta SQL em um banco de dados de voos."""
    req_keys = ['first_name', 'last_name', 'confirmation']
    
    # Se faltar algum dado essencial, avisa que a busca não pode ser feita
    if not all(d.get(key) not in ['unknown', None, -1, ''] for key in req_keys):
        return "Faltam informações para consultar o banco de dados. Peça o nome, sobrenome e número de confirmação (5 dígitos)."

    keys = req_keys + ["departure", "destination", "departure_time", "arrival_time", "flight_day"]
    values = [
        ["Jane", "Doe", 12345, "San Jose", "New Orleans", "12:30 PM", "9:30 PM", "tomorrow"],
        ["John", "Smith", 54321, "New York", "Los Angeles", "8:00 AM", "11:00 AM", "Sunday"],
    ]
    get_key = lambda d: "|".join([str(d.get('first_name')), str(d.get('last_name')), str(d.get('confirmation'))])
    get_val = lambda l: {k:v for k,v in zip(keys, l)}
    db = {get_key(get_val(entry)) : get_val(entry) for entry in values}

    data = db.get(get_key(d))
    if not data:
        return f"Nenhum voo encontrado para {d['first_name']} {d['last_name']} com a confirmação {d['confirmation']}."
        
    return (f"O voo de {data['first_name']} {data['last_name']} de {data['departure']} para {data['destination']} "
            f"parte às {data['departure_time']} {data['flight_day']} e pousa às {data['arrival_time']}.")

# 3. Definição da Estrutura de Memória (Pydantic)
class KnowledgeBase(BaseModel):
    first_name: str = Field('unknown', description="Primeiro nome do usuário. 'unknown' se não souber.")
    last_name: str = Field('unknown', description="Sobrenome do usuário. 'unknown' se não souber.")
    confirmation: Optional[int] = Field(-1, description="Número de confirmação de 5 dígitos. -1 se não souber.")
    discussion_summary: str = Field("", description="Resumo breve da conversa.")
    current_goals: str = Field("", description="O que o usuário quer saber?")

# 4. A Função "Mágica" de Extração (Fornecida no curso)
def RExtract(pydantic_class, llm, prompt):
    '''Força o LLM a cuspir um JSON que preencha a nossa classe Pydantic'''
    parser = PydanticOutputParser(pydantic_object=pydantic_class)
    instruct_merge = RunnableAssign({'format_instructions' : lambda x: parser.get_format_instructions()})
    
    def preparse(string):
        """Limpa a saída do LLM caso ele adicione textos antes do JSON (comum em modelos locais)"""
        if '{' in string: string = string[string.find('{'):]
        if '}' in string: string = string[:string.rfind('}')+1]
        return string
        
    return instruct_merge | prompt | llm | StrOutputParser() | RunnableLambda(preparse) | parser

# 5. Prompts
parser_prompt = ChatPromptTemplate.from_template(
    "Você é uma IA de extração de dados. Extraia as informações da mensagem do usuário para atualizar a base de conhecimento.\n"
    "{format_instructions}\n\n"
    "BASE DE CONHECIMENTO ANTIGA: {know_base}\n"
    "MENSAGEM DO USUÁRIO: {input}\n\n"
    "NOVA BASE DE CONHECIMENTO (Apenas o JSON validado):"
)

external_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Você é o agente de atendimento da SkyFlow Airlines. Seja amigável e conciso.\n"
        "Use o contexto abaixo para responder ao usuário. Se o contexto disser que faltam dados, "
        "peça educadamente APENAS as informações que faltam (nome, sobrenome ou confirmação).\n"
        "CONTEXTO DO BANCO DE DADOS: {context}\n"
    )),
    ("user", "{input}")
])

# 6. RESOLUÇÃO DOS TODOs: Construindo a engrenagem interna
# Extrai o JSON e substitui o 'know_base' no nosso dicionário de estado
extractor = RExtract(KnowledgeBase, instruct_llm, parser_prompt)

# Função auxiliar para consultar o DB usando a base de conhecimento atualizada
def consultar_db(estado):
    kb = estado['know_base']
    # Transforma o objeto Pydantic em um dicionário para a função do banco de dados
    dados_busca = {'first_name': kb.first_name, 'last_name': kb.last_name, 'confirmation': kb.confirmation}
    return get_flight_info(dados_busca)

# A Cadeia Interna de Estado (State Chain)
internal_chain = (
    RunnableAssign({'know_base': extractor}) 
    | RunnableAssign({'context': RunnableLambda(consultar_db)})
)

external_chain = external_prompt | chat_llm | StrOutputParser()

# 7. Orquestrador do Chat (Adaptado para seu ambiente Gradio)
# Criamos um estado global inicial
estado_global = {'know_base': KnowledgeBase()}

def chat_gen(message, history):
    global estado_global
    
    # Injeta a nova mensagem no estado
    estado_global['input'] = message
    
    print("\n[🧠 PENSANDO] Extraindo dados e consultando o banco...")
    # Roda a cadeia interna (atualiza a memória e consulta o DB)
    estado_global = internal_chain.invoke(estado_global)
    
    print(f"[🔍 DEBUG] Memória Atualizada: {estado_global['know_base'].model_dump()}")
    print(f"[📊 DEBUG] Retorno do DB: {estado_global['context']}")
    
    # Roda a cadeia externa (fala com o usuário) passando o estado enriquecido
    buffer = ""
    for token in external_chain.stream(estado_global):
        buffer += token
        yield buffer

# 8. Interface Gradio
if __name__ == "__main__":
    chat_history = [{"role": "assistant", "content": "Olá! Sou o agente da SkyFlow Airlines. Como posso ajudar com seu voo hoje?"}]
    # Removido o type="messages" para manter a compatibilidade com a sua versão do Gradio
    chatbot = gr.Chatbot(value=chat_history)
    demo = gr.ChatInterface(chat_gen, chatbot=chatbot)
    demo.launch()