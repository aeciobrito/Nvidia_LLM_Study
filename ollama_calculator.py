import json

def obter_ranking_velocidade(velocidade):
    """
    Retorna uma classificação baseada na velocidade em tokens/segundo,
    utilizando o consenso da comunidade sobre fluidez e leitura.
    """
    if velocidade == 0:
        return "❌ Falha (Nenhum token gerado)"
    elif velocidade < 10:
        return "🐢 Lento (Abaixo da velocidade natural de leitura)"
    elif velocidade < 25:
        return "🚶 Bom (Confortável para leitura em tempo real)"
    elif velocidade < 50:
        return "🏃 Excelente (Mais rápido que leitura dinâmica)"
    else:
        return "⚡ Fantástico (Velocidade extrema, ideal para Agentes Autônomos)"
    
def analisar_performance(metadata):
    """
    Recebe o dicionário de response_metadata do LangChain/Ollama 
    e exibe um relatório de performance amigável com ranking.
    """
    # 1. Extraindo os dados brutos
    modelo = metadata.get('model_name', 'Desconhecido')
    tokens_gerados = metadata.get('eval_count', 0)
    tempo_geracao_ns = metadata.get('eval_duration', 0)
    
    tokens_prompt = metadata.get('prompt_eval_count', 0)
    tempo_prompt_ns = metadata.get('prompt_eval_duration', 0)
    tempo_total_ns = metadata.get('total_duration', 0)

    # 2. Conversões (Nanossegundos para Segundos)
    tempo_geracao_s = tempo_geracao_ns / 1e9
    tempo_prompt_s = tempo_prompt_ns / 1e9
    tempo_total_s = tempo_total_ns / 1e9

    # 3. Cálculos
    velocidade = (tokens_gerados / tempo_geracao_s) if tempo_geracao_s > 0 else 0
    ranking = obter_ranking_velocidade(velocidade)

    # 4. Exibição do Relatório
    print("=" * 50)
    print(f"📊 RELATÓRIO DE PERFORMANCE: {modelo.upper()}")
    print("=" * 50)
    print(f"🚀 Velocidade (Tokens/s) : {velocidade:.2f} t/s")
    print(f"⭐ Classificação         : {ranking}")
    print("-" * 50)
    print(f"🧠 Leitura do Prompt     : {tokens_prompt} tokens em {tempo_prompt_s:.3f}s")
    print(f"✍️ Geração da Resposta   : {tokens_gerados} tokens em {tempo_geracao_s:.2f}s")
    print(f"⏱️ Tempo Total Gasto     : {tempo_total_s:.2f}s")
    print("=" * 50)

if __name__ == "__main__":
    # Cole os dados de AIMessage.response_metadata
    meus_dados = {'model': 'gemma4:26b', 'created_at': '2026-04-25T19:08:14.894857338Z', 'done': True, 'done_reason': 'stop', 'total_duration': 114402309925, 'load_duration': 3689997226, 'prompt_eval_count': 43, 'prompt_eval_duration': 374276171, 'eval_count': 2161, 'eval_duration': 109183539778, 'logprobs': None, 'model_name': 'gemma4:26b', 'model_provider': 'ollama'}
    analisar_performance(meus_dados)