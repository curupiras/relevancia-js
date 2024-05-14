import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def precisao_recall(docs_retornados, docs_relevantes, k=None):
    """
        Dado um conjunto de documentos retornados e de documentos relevantes,
        calcula a precisão e o recall em k para uma query.

        docs_retornados -- Objeto Series contendo os documentos retornados ordenados
        docs_relevantes -- Objeto Series contendo os documentos relevantes
        k -- No cálculo da precisão e recall, indica até que posição dos documentos
                retornados deve ser considerada.

        Se k = None, toda a lista de documentos retornados é considerada. Se k != None,
        considera apenas os k'éssimos primeiros documentos retornados.
        Para o cálculo da precisão, se k = None, considera no denominador o total de documentos retornados
    """
    if k is None:
        k = len(docs_retornados)
    docs_retornados_em_k = docs_retornados[:k]
    docs_retornados_em_k_relevantes = set(docs_retornados_em_k) & set(docs_relevantes)

    precisao = len(docs_retornados_em_k_relevantes)/max(k, 1)
    recall = len(docs_retornados_em_k_relevantes)/len(docs_relevantes)

    return precisao, recall

def mrr(docs_retornados, docs_relevantes, k=None):
    """
        Calcula o MRR@k (Mean Reciprocal Rank) para uma query.

        docs_retornados -- Objeto Series contendo os documentos retornados ordenados
        docs_relevantes -- Objeto Series contendo os documentos relevantes
        k -- Indica até que posição dos documentos retornados deve ser considerada.
    """
    if k is None:
        k = len(docs_retornados)

    mrr_score = 0.0
    set_docs_relevantes = set(docs_relevantes)
    for i in range(min(k, len(docs_retornados))):
        if docs_retornados.iloc[i] in set_docs_relevantes:
            mrr_score = 1.0 / (i+1) # Soma com 1 pois a posição começa em 1 e i começa em 0.
            break
    return mrr_score

def dcg(doc_retornados, doc_relevantes, k=None, debug=True, aproximacao_trec_eval=False):
    """
        Calcula DCG@k para uma query.

        doc_retornados -- É uma lista de keys de documentos. A posição do documento na lista
            indica a ordem
        docs_relevantes -- É um dict cuja chave é a key e um documento relevante e o valor
            é o seu score
        k -- Indica até que posição dos documentos retornados deve ser considerada.
        debug -- Indica se é pra imprimir o cálculo intermediário
        aproximacao_trec_eval -- Se True, usa a relevância como Linear. Se False, usa
            como 2^(rel)
    """
    dcg = 0
    doc_retornados = doc_retornados if k is None else doc_retornados[:k]
    for rank, doc_id in enumerate(doc_retornados, 1):
        # Relevância do documento
        rel = doc_relevantes.get(doc_id, 0)
        # Cálculo do ganho. Aproximação trec_eval usa diretamente a relevância
        gain = (2**(rel) - 1) if not aproximacao_trec_eval else rel
        dcg_i = gain/(math.log(rank + 1, 2))
        dcg += dcg_i
        if debug:
            print(doc_id, rank, dcg_i)

    if debug:
        print('\n')
    return dcg

def idcg(doc_retornados, doc_relevantes, k=None, debug=True, aproximacao_trec_eval=False):
    """
        Calcula iDCG@k para uma query.

        doc_retornados -- É uma lista de keys de documentos. A posição do documento na lista
            indica a ordem
        docs_relevantes -- É um dict cuja chave é a key e um documento relevante e o valor
            é o seu score
        k -- Indica até que posição dos documentos retornados deve ser considerada.
        debug -- Indica se é pra imprimir o cálculo intermediário
        aproximacao_trec_eval -- Se True, usa a relevância como Linear. Se False, usa
            como 2^(rel)
    """
    # Cria uma lista de tuplas (doc_id, relevância, posição original na lista de retornados)
    # para todos os documentos relevantes
    # A posição original só é usada para desempate, portanto, ela segue a ordem de doc_retornados
    docs_com_relevancia = [
        (doc, 
        doc_relevantes.get(doc, 0), # Nem precisava de get, pois certamente existe
        doc_retornados.index(doc) if doc in doc_retornados else len(doc_retornados))
        for doc in doc_relevantes.keys()
    ]

    # Ordena os documentos primeiro pela relevância (decrescente) e depois pela posição original (crescente)
    # Isso garante que, em caso de empate na relevância, o documento que apareceu primeiro em doc_retornados ganhe
    docs_ordenados = sorted(docs_com_relevancia, key=lambda x: (-x[1], x[2]))

    # Extrai apenas os doc_ids da lista ordenada
    doc_retornados_ideal = [doc[0] for doc in docs_ordenados]

    return dcg(doc_retornados_ideal, doc_relevantes, k, debug, aproximacao_trec_eval)

def ndcg(resultado_pesquisa, qrels, col_resultado_doc_key, col_qrels_doc_key, col_qrels_score, k=None, debug=True, aproximacao_trec_eval=False):
    """
        Calcula o nDCG@k para uma query

        resultado_pesquisa -- DataFrame Pandas com o resultado da pesquisa. Considera que
            o DataFrame está ordenado de acordo com os documentos retornados
        qrels -- DataFrame Pandas com o qrels

        col_resultado_doc_key -- indica a KEY do documento retornado.
        col_qrels_doc_key -- indica a KEY de um documento associado a query.
        col_qrels_score -- indica a relevância do documento para aquela query. Quanto maior, mais relevante.

        k -- Indica até que posição dos documentos retornados deve ser considerada.
        debug -- Indica se é pra imprimir o cálculo intermediário
        aproximacao_trec_eval -- Se True, usa a relevância como Linear. Se False, usa
            como 2^(rel)
    """
    # Converte os pandas para lista de doc_retornados e dict de doc_relevantes por score:
    doc_retornados = resultado_pesquisa[col_resultado_doc_key].tolist()
    doc_relevantes = dict(zip(qrels[col_qrels_doc_key], qrels[col_qrels_score]))

    return dcg(doc_retornados, doc_relevantes, k, debug, aproximacao_trec_eval) / idcg(doc_retornados, doc_relevantes, k, debug, aproximacao_trec_eval)

def metricas(resultado_pesquisa, qrels, 
             col_resultado_query_key="QUERY_KEY",
             col_resultado_doc_key="DOC_KEY",
             col_resultado_rank="RANK",
             col_qrels_query_key="QUERY_KEY",
             col_qrels_doc_key="DOC_KEY",
             col_qrels_score="SCORE",
             k=[5, 10, 20, 50], debug=False, aproximacao_trec_eval=False):
    """
        Calcula um conjunto de métricas para um resultado de pesquisa e um conjunto qrels.
        resultado_pesquisa -- DataFrame Pandas contendo o resultado das pesquisas.
        qrels -- DataFrame Pandas contendo o qrels

        Os parâmetros col_resultado_xxxx referem-se a nomes de colunas no DataFrame resultado_pesquisa:

        col_resultado_query_key -- indica a KEY da query.
        col_resultado_doc_key -- indica a KEY do documento retornado.
        col_resultado_rank -- indica a posição do documento retornado.

        Os parâmetros col_qrels_xxxx referem-se a nomes de colunas no DataFrame qrels:

        col_qrels_query_key -- indica a KEY da query que será testada.
        col_qrels_doc_key -- indica a KEY de um documento associado a query.
        col_qrels_score -- indica a relevância do documento para aquela query. Quanto maior, mais relevante.
    """
    # Remove do qrels os resultados cujo score é 0
    qrels = qrels[qrels[col_qrels_score] > 0]

    # Extrai as queries que devem ser analisadas. Se tiver query no resultado que não 
    # está no qrels, ela não será avaliada.
    query_keys = qrels.QUERY_KEY.unique()

    precisao_em_k = {valor_k: [0]*len(query_keys) for valor_k in k}
    recall_em_k = {valor_k: [0]*len(query_keys) for valor_k in k}
    mrr_em_k = {valor_k: [0]*len(query_keys) for valor_k in k}
    ndcg_em_k = {valor_k: [0]*len(query_keys) for valor_k in k}

    for i_q_key, q_key in enumerate(query_keys):
        # Extrai o resultado e o qrels para a query que irá ser analisada
        resultado_para_query = resultado_pesquisa[resultado_pesquisa[col_resultado_query_key] == q_key]
        qrels_para_query = qrels[qrels[col_qrels_query_key] == q_key]
        
        # Pega os docs retornados (ordenados de acordo com a posição deles na pesquisa, em ordem crescente - Rank 1 para cima)
        # e os docs relevantes.
        resultado_para_query = resultado_para_query.sort_values(by=col_resultado_rank)
        docs_retornados = resultado_para_query[col_resultado_doc_key]
        docs_relevantes = qrels_para_query[col_qrels_doc_key]

        for valor_k in k:
            p_em_k, r_em_k = precisao_recall(docs_retornados, docs_relevantes, valor_k)
            precisao_em_k[valor_k][i_q_key] = p_em_k
            recall_em_k[valor_k][i_q_key] = r_em_k
            mrr_em_k[valor_k][i_q_key] = mrr(docs_retornados, docs_relevantes, valor_k)
            ndcg_em_k[valor_k][i_q_key] = ndcg(resultado_para_query, qrels_para_query, col_resultado_doc_key, col_qrels_doc_key, col_qrels_score, valor_k, debug, aproximacao_trec_eval)

    pd_metricas = pd.DataFrame({'QUERY_KEY': query_keys})

    # Insere as métricas na ordem: precisão, recall, MRR, nDCG:
    for valor_k in k:
        pd_metricas[f'P@{valor_k}'] = precisao_em_k[valor_k]
    for valor_k in k:
        pd_metricas[f'R@{valor_k}'] = recall_em_k[valor_k]
    for valor_k in k:
        pd_metricas[f'MRR@{valor_k}'] = mrr_em_k[valor_k]
    for valor_k in k:
        pd_metricas[f'nDCG@{valor_k}'] = ndcg_em_k[valor_k]

    return pd_metricas



# Funções para plotar métricas
def histograma_metricas(df, metrica_1='P@10', metrica_2='R@10', metrica_3='MRR@10', metrica_4='nDCG@10', ylim=None, bins=10):
    plt.figure(figsize=(12, 8))

    cores_seaborn = sns.color_palette('deep')
    alpha = 1
    
    plt.subplot(2, 2, 1)
    plt.hist(df[metrica_1], bins=np.linspace(0, 1, bins+1), color=cores_seaborn[0], alpha=alpha)
    plt.title(f'Histograma de {metrica_1}')
    plt.xlabel('Valor')
    plt.ylabel('Frequência')
    plt.xlim(0, 1)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    
    plt.subplot(2, 2, 2)
    plt.hist(df[metrica_2], bins=np.linspace(0, 1, bins+1), color=cores_seaborn[1], alpha=alpha)
    plt.title(f'Histograma de {metrica_2}')
    plt.xlabel('Valor')
    plt.ylabel('Frequência')
    plt.xlim(0, 1)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    
    plt.subplot(2, 2, 3)
    plt.hist(df[metrica_3], bins=np.linspace(0, 1, bins+1), color=cores_seaborn[2], alpha=alpha)
    plt.title(f'Histograma de {metrica_3}')
    plt.xlabel('Valor')
    plt.ylabel('Frequência')
    plt.xlim(0, 1)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    
    plt.subplot(2, 2, 4)
    plt.hist(df[metrica_4], bins=np.linspace(0, 1, bins+1), color=cores_seaborn[3], alpha=alpha)
    plt.title(f'Histograma de {metrica_4}')
    plt.xlabel('Valor')
    plt.ylabel('Frequência')
    plt.xlim(0, 1)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    
    plt.tight_layout()
    plt.show()

def boxplot_metricas(df, metricas=['P@10', 'R@10', 'MRR@10', 'nDCG@10']):
    df_melted = df.melt(value_vars=metricas, var_name='metric', value_name='value')

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='metric', y='value', data=df_melted)
    plt.xticks(rotation=45)
    plt.xlabel('Métrica')
    plt.ylabel('Valor')
    plt.title('Boxplot das Métricas')
    plt.show()