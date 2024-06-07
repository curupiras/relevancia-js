from collections import Counter
from tqdm import tqdm
import array
import math
import pickle
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from unidecode import unidecode
import string

from formatador import remove_html

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

def tokenizador_pt(texto):
    # Remove acentuação e converte para minúsculo
    texto = unidecode(texto.lower())
    
    # Remove pontuação
    texto = ''.join([char if char not in string.punctuation else ' ' for char in texto])
    
    # Tokeniza o texto
    tokens = word_tokenize(texto, language='portuguese')
    
    # Remove stopwords e aplica stemização
    stemmer = RSLPStemmer()
    tokens_processados = [stemmer.stem(token) for token in tokens if token not in stopwords.words('portuguese')]
    
    return tokens_processados

def tokenizador_pt_remove_html(texto):
    return tokenizador_pt(remove_html(texto))

# Definição de uma classe para índice invertido
class IndiceInvertido:

    # Recebe 'tokenizar', uma função tokenizadora. Por padrão, é a função tokenizador_pt
    def __init__(self, tokenizar=tokenizador_pt):
        # Cria um índice invertido vazio.
        # A chave é um token e o valor é um objeto contendo a lista de ids de documento que contém o token
        # e uma lista de total de ocorrências do token naquele documento
        self.indice = {}
        # Cria um índice de tamanho de documentos vazio.
        # A chave é a id do documento e o valor é o total de tokens do documento
        self.tamanho_doc = {}
        # Guarda o total de documentos adicionados. É o mesmo que len(self.tamanho_doc.keys())
        self.n_docs = 0
        # Tokenizador. Essa função é aplicada para a extração dos tokens 
        # de um documento. Ela deve fazer todo o pré-processamento (conversão para minúsculo,
        # remoção de stop words, lematização etc)
        self.tokenizar = tokenizar

    def adiciona_dataframe(self, df, extrai_id_conteudo_row):
        for index, row in tqdm(df.iterrows(), total=len(df)):
            id_doc, conteudo_doc = extrai_id_conteudo_row(row)
            self.adiciona_doc(id_doc, conteudo_doc)
            
    def adiciona_objetos(self, lista_objetos, extrai_id_conteudo):
        for objeto in tqdm(lista_objetos):
            id_doc, conteudo_doc = extrai_id_conteudo(objeto)
            self.adiciona_doc(id_doc, conteudo_doc)
        
    def adiciona_doc(self, id_doc, conteudo_doc=""):
        # Extrai os tokens de um documento
        tokens = self.tokenizar(conteudo_doc)
        # Conta quantas vezes cada token aparece no documento
        contador_tokens_no_documento = Counter(tokens)

        for token, n_ocorrencias in contador_tokens_no_documento.items():
            # Se o token ainda não está no índice, cria
            self.indice.setdefault(token, {"id_doc": [], "n_ocorrencias": array.array("L", [])})
            # Popula as informações do token para esse documento (a id do documento e
            # quantas vezes o token apareceu no documento)
            self.indice[token]['id_doc'].append(id_doc)
            self.indice[token]['n_ocorrencias'].append(n_ocorrencias)

        # Adiciona um documento a n_docs
        self.n_docs += 1
        # Salva o tamanho do documento
        self.tamanho_doc[id_doc] = len(tokens)

    # Funções utilitárias
    def get_tamanho_medio_docs(self):
        # Utilitário para calcular o tamanho médio dos documentos no índice
        return sum(self.tamanho_doc.values()) / self.n_docs
    
    def total_docs_por_token(self, token):
        # Utilitário para retornar o total de documentos que contém o token
        return len(self.indice[token]['id_doc'])
        
    def to_pickle(self, nome_arquivo):
        # Para salvar o índice num arquivo, precisamos manter
        # self.indice, self.tamanho_doc e self.n_docs
        obj_para_salvar = {"indice": self.indice, "tamanho_doc": self.tamanho_doc, "n_docs": self.n_docs}
        with open(nome_arquivo, 'wb') as f:
            pickle.dump(obj_para_salvar, f)

    def from_pickle(self, nome_arquivo):
        # Fazemos o caminho inverso aqui:
        with open(nome_arquivo, 'rb') as f:
            obj_recuperado = pickle.load(f)
        self.indice = obj_recuperado['indice']
        self.tamanho_doc = obj_recuperado['tamanho_doc']
        self.n_docs = obj_recuperado['n_docs']
        
class BM25:
    # A consideração que é feita aqui é que primeiro é criado um índice invertido
    # e populado e, depois, é instanciado um BM25 com esse índice invertido.
    # A ideia é que o indiceInvertido mantém só a relação de tokens e documentos.
    # O BM25 é que faz o cálculo do score. Assim, um indice invertido
    # pode ser compartilhado por mais de um BM25 com configurações distintas
    # (útil para testar diferentes k1 e b, por exemplo, sem ter que reindexar tudo)
    def __init__(self, indice_invertido=IndiceInvertido(), k1 = 0.9, b = 0.4, bias_idf = 0):
        self.bias_idf = bias_idf
        self.k1 = k1
        self.b = b
        # Para setar o índice é necessário já ter a informação
        self.set_indice_invertido(indice_invertido)

    def set_params(self, k1=None, b=None, bias_idf=None):
        # Se mudar qualquer parâmetro, tem que reiniciar o score
        # Se mudar o bias_idf, é necessário recalcular o idf
        if k1 is not None:
            self.k1 = k1
        if b is not None:
            self.b = b
        if bias_idf is not None:
            self.bias_idf = bias_idf
            self.precalcula_idf()        

        self.reinicia_score_dos_indices()
    
    def set_indice_invertido(self, indice_invertido):
        # Quando é adicionado um índice invertido, é necessário reiniciar todo o score/idf calculado
        self.indice_invertido = indice_invertido
        # Guarda o tamanho médio dos documentos do índice
        self.avgdl = self.indice_invertido.get_tamanho_medio_docs()
        # Pré-calcula o idf de cada token
        self.precalcula_idf()
        self.reinicia_score_dos_indices()

    def reinicia_score_dos_indices(self):
        self.score_por_token = {}

    def precalcula_idf(self):
        # Vamos criar um mapa para guardar os idfs pré-calculados para cada token
        self.idf_por_token = {}

        # Número de documento do corpus está presente no objeto indice_invertido
        N = self.indice_invertido.n_docs

        # Varre todos os tokens do índice. Os tokens são as chaves do indice_invertido.indice
        for token in self.indice_invertido.indice.keys():
            # Recupera o total de documentos que tem o token
            n_doc_token = self.indice_invertido.total_docs_por_token(token)        
            
            # Isso já é o suficiente pra calcular o idf (math.log é o logaritmo na base e)
            idf_token = math.log( ((N - n_doc_token + 0.5)/(n_doc_token + 0.5)) + self.bias_idf )
            # E agora, vamos guardar essa informação
            self.idf_por_token[token] = idf_token

    def calcula_score_para_um_token_e_salva(self, token):
        # O cálculo do BM25 para determinada query é a multiplicação do idf pela frequência do termo no documento * (k1 + 1)
        # Além disso, é dividido pela frequencia do termo no documento + k1 * (1 - b + b * tamanho_doc/avgdl)
        idf = self.idf_por_token[token]
        # Juntando tudo, podemos calcular o score pelo BM25
        zip_id_freq = zip(self.indice_invertido.indice[token]['id_doc'], self.indice_invertido.indice[token]['n_ocorrencias'])
        bm25 = array.array("f", [ idf * freq_token_no_doc * (self.k1 + 1) / (freq_token_no_doc + self.k1 * (1 - self.b + self.b * self.indice_invertido.tamanho_doc[id_doc] / self.avgdl)) for (id_doc, freq_token_no_doc) in zip_id_freq ])
        # Salva o bm25 no índice
        self.score_por_token[token] = bm25

    def tokenizar(self, query):
        return self.indice_invertido.tokenizar(query)

    def pesquisar(self, query):
        # Tokeniza a query
        tokens = self.tokenizar(query)
    
        # Se não tem token para ser pesquisado, retorna conjunto vazio
        if (len(tokens) == 0):
            return []

        # Guarda um dicionário onde a chave é o id do documento e o valor é o score desse documento para a query pesquisada
        docs_retornado_com_score = Counter({})

        # Faz a pesquisa de documentos. Para isso iteramos todos os tokens da query
        for token in tokens:
            # É possível que a query contenha algum termo que não foi indexado. Se isso ocorrer,
            # entende-se que a frequência desse token em qualquer documento é 0, já que não pode ser encontrado
            if token not in self.indice_invertido.indice:
                continue

            # Pega a lista de documentos que será analisado
            docs_que_tem_token = self.indice_invertido.indice[token]['id_doc']

            # Se for a primeira vez que esse token é pesquisado, é necessário calcular o score relacionado
            # a ele e salvar. Se já tiver sido feito antes, já podemos buscar o cálculo pronto (que funciona
            # como um cache. Isso é útil no caso de várias pesquisas seguidas)
            if token not in self.score_por_token.keys():
                self.calcula_score_para_um_token_e_salva(token)
            score_dos_docs_deste_token = self.score_por_token[token]

            # Agora já temos calculado o score de todos os documentos desse token. Só adiciona ao acumulador de score atual
            # docs_retornado_com_score += score_dos_docs_deste_token -> Se fosse usar dict direto no índice seria assim, mas a memória não está aguentando guardar os scores de ambos
            for id_doc, score_par_doc_token in zip(docs_que_tem_token, score_dos_docs_deste_token):
                docs_retornado_com_score[id_doc] += score_par_doc_token

        # Agora converte esse dict em uma lista de tuplas com a chave (id_doc) e valor (score_do_doc)
        docs_com_score = list(docs_retornado_com_score.items())

        # E ordena do mais relevante para o menos relevante
        return sorted(docs_com_score, key=lambda x: x[1], reverse=True)