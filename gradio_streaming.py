import gradio as gr
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Configuração do LLM
# Instancia a conexão com o modelo local.
instruct_llm = ChatOllama(model="gemma4:latest", temperature=0.9)

# 2. Criação dos Templates de Prompt
# Templates definem a estrutura da requisição e preparam os "espaços" para injeção de variáveis (input, topic).
prompt1 = ChatPromptTemplate.from_messages([
    ("user", "INSTRUCAO: Responda em rimas\n\nPROMPT: {input}")    
])

prompt2 = ChatPromptTemplate.from_messages([
    ("user", "INSTRUCAO: Responda em rimas, mude o topico do poema para {topic}!\n"
    "Deixe ele feliz! Tente manter a mesma estrutura das sentenças, mas faça-as fáceis de recitar!\n\n"
    "Poema original: {input}\n\nNovo topico: {topic}")
])

# 3. Construção das Chains (LangChain Expression Language - LCEL)
# O operador '|' encadeia o Template, o LLM e o Parser em um único pipeline de execução.
chain1 = prompt1 | instruct_llm | StrOutputParser()
chain2 = prompt2 | instruct_llm | StrOutputParser()

def rhyme_chat2_stream(message, history):
    """
    Função orquestradora. Avalia o contexto da conversa e decide (Routing)
    qual pipeline do LangChain deve ser acionado.
    """
    first_poem = None
    
    # 4. Extração de Contexto (Parse do Histórico)
    # Varre as mensagens anteriores do Gradio para tentar resgatar o poema base.
    for entry in history:
        if isinstance(entry, dict) and entry.get("role") == "assistant":
            lista_conteudos = entry.get("content", [])
            
            if len(lista_conteudos) > 0:
                texto_puro = lista_conteudos[0].get("text", "")
                
                # Marcadores para localizar e recortar exatamente o texto do poema gerado
                marcador_inicio = "Deixe-me pensar!"
                marcador_fim = "Agora, me deixe reescrever"
                
                if marcador_inicio in texto_puro and marcador_fim in texto_puro:
                    inicio = texto_puro.find(marcador_inicio) + len(marcador_inicio)
                    fim = texto_puro.find(marcador_fim)
                    
                    if fim > inicio:
                        first_poem = texto_puro[inicio:fim].strip()
                        break

    # 5. Roteamento Lógico e Execução de Streaming
    if first_poem is None:
        # ROTA A: Criação inicial. Aciona a Chain 1 passando apenas o input.
        buffer = "Oh! Eu consigo fazer um belo poema sobre isso! Deixe-me pensar!\n\n"
        yield buffer
        
        # Utilizamos .stream() em vez de .invoke() para enviar a resposta token a token para a UI
        chat_gen = chain1.stream({"input": message})
        for token in chat_gen:
            buffer += token
            yield buffer

        passage = "\n\nAgora, me deixe reescrever com um foco diferente. Qual deveria ser o foco?"
        buffer += passage
        yield buffer
        
    else:
        # ROTA B: Reescrita. Aciona a Chain 2 passando o poema base encontrado no histórico e o novo tópico.
        buffer = "Ótimo! Lá vamos nós!\n\n"
        yield buffer

        chat_gen = chain2.stream({"input": first_poem, "topic": message})
        for token in chat_gen:
            buffer += token
            yield buffer
        
        passage = "\n\nE aí, o que achou? Quer tentar outro tópico?"
        buffer += passage
        yield buffer

# 6. Inicialização da Interface Gráfica (Gradio)
if __name__ == "__main__":
    # Preenche o chatbot com uma mensagem inicial para ditar a primeira interação
    saudacao_inicial = [{"role": "assistant", "content": "Deixe-me ajudar a criar um poema! Sobre o que você gostaria que fosse?"}]
    chatbot = gr.Chatbot(value=saudacao_inicial)
    
    demo = gr.ChatInterface(rhyme_chat2_stream, chatbot=chatbot)
    demo.launch()