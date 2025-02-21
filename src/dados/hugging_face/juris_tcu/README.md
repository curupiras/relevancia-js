# JurisTCU: A Brazilian Portuguese Information Retrieval Dataset with Query Relevance Judgments

## Overview

JurisTCU is a dataset for Legal Information Retrieval (LIR) in Brazilian Portuguese. It consists of jurisprudence from the Brazilian Federal Court of Accounts (Tribunal de Contas da União – TCU) and provides query relevance judgments (qrels) to support the evaluation and improvement of legal search systems.

The dataset includes:
- **16,045 legal documents** from the curated "Selected Jurisprudence" collection of the TCU.
- **150 standardized queries**, divided into three categories based on real user interactions and synthetic query generation.
- **2,250 relevance judgments**, with each query associated with 15 manually reviewed document assessments.

## Motivation

Legal professionals, policymakers, researchers, and citizens rely on efficient search tools to navigate extensive legal texts. However, the retrieval of legal documents presents unique challenges, such as complex language, evolving jurisprudence, and vocabulary mismatch.

JurisTCU addresses these challenges by providing a **qrels-based benchmark** for evaluating legal search engines in Brazilian Portuguese. It enables researchers to test and compare different retrieval approaches, including **lexical (BM25) and semantic (Transformer-based embeddings) search techniques**.

## Usage
The dataset can be used to evaluate and benchmark IR systems using both lexical and semantic search techniques. It is particularly useful for researchers focusing on the Portuguese language and legal information retrieval.

### Loading the dataset

```python
from datasets import load_dataset

dataset = load_dataset("Leandro/JurisTCU")
documents = dataset["documents"]
queries = dataset["queries"]
qrels = dataset["qrels"]
```

## Dataset Structure

The JurisTCU dataset consists of three main files:

```plaintext
JurisTCU/
│── doc.csv        # Collection of legal documents from the TCU
│── query.csv      # Set of 150 standardized queries
│── qrel.csv       # Relevance judgments for each query
```


### **Documents (`doc.csv`)**
The `doc.csv` file contains **16,045 legal documents** from the TCU's selected jurisprudence. Each row represents a legal document and includes the following fields:

#### Fields of a Document in the Corpus

| **Field**                    | **Description**                          | **Properties**                        |
|------------------------------|------------------------------------------|---------------------------------------|
| `KEY`                        | Identifier Key                           | Text                                  |
| `NUMACORDAO`                 | Decision number                          | Number                                |
| `ANOACORDAO`                 | Decision year                            | Number                                |
| `COLEGIADO`                  | Collegiate                               | Text                                  |
| `AREA`                       | Area indexer                             | Text (e.g., personnel, bidding)      |
| `TEMA`                       | Topic indexer                            | Text (e.g., debt, fifths)            |
| `SUBTEMA`                    | Subtopic indexer                         | Text (e.g., earnings, prohibition)   |
| `ENUNCIADO`                  | Summary                                  | Text                                  |
| `EXCERTO`                    | Document excerpt                         | Text                                  |
| `NUMSUMULA`                  | Legal summary (precedent) number         | Number                                |
| `DATASESSAOFORMATADA`        | Date of the judgment session             | Date (DD/MM/YYYY)                    |
| `AUTORTESE`                  | Author of the legal thesis               | Text                                  |
| `FUNCAOAUTORTESE`            | Role of the author                       | Text                                  |
| `TIPOPROCESSO`               | Type of the process                      | Text (e.g., denunciation, accounts)  |
| `TIPORECURSO`                | Type of the appeal                       | Text (e.g., review request)          |
| `INDEXACAO`                  | Generic indexers                         | Text (e.g., requirement)             |
| `INDEXADORESCONSOLIDADOS`    | All Indexers                             | Text                                  |
| `PARAGRAFOLC`                | Paragraph on bidding and contracts       | Text                                  |
| `REFERENCIALEGAL`            | Legal reference                          | Text                                  |
| `PUBLICACAOAPRESENTACAO`     | URL of the publication                   | URL                                   |
| `PARADIGMATICO`              | Paradigmatic type indexer                | Text (e.g., consultation)            |


#### Example document entry
```json
{
  "KEY": "JURISPRUDENCIA-SELECIONADA-85434",
  "NUMACORDAO": 3580,
  "ANOACORDAO": 20200,
  "COLEGIADO": "Plenário",
  "AREA": "Finanças Públicas",
  "TEMA": "Regime Próprio de Previdência Social",
  "SUBTEMA": "Pensão",
  "ENUNCIADO": "SÚMULA TCU 43 (REVOGADA): As pensões deferidas antes de 21/10/69, aos dependentes do pessoal, reformado, ou em atividade, da Polícia Militar e do Corpo de Bombeiros, transferido para o Estado da Guanabara, devem ser custeadas pela União, cabendo, porém, ao referido Estado a responsabilidade integral do pagamento decorrente dos reajustamentos posteriores.",
  "EXCERTO": "Relatório: Trata-se de estudo elaborado pela Secretaria das Sessões (Seses), para avaliar a utilidade e a pertinência dos enunciados da súmula de jurisprudência do Tribunal de Contas da União referentes aos grupos temáticos denominados 'Estado da Guanabara' e 'Fundos de Participação'...",
  "NUMSUMULA": 43,
  "DATASESSAOFORMATADA": "19/02/2020",
  "AUTORTESE": "RAIMUNDO CARREIRO",
  "FUNCAOAUTORTESE": "RELATOR",
  "TIPOPROCESSO": "ADMINISTRATIVO",
  "TIPORECURSO": null,
  "INDEXACAO": ["Súmula", "Guanabara", "Custeio", "Bombeiro militar", "Polícia Militar", "Pensão militar"],
  "INDEXADORESCONSOLIDADOS": "AREA: Finanças Públicas ; TEMA: Regime Próprio de Previdência Social ; SUBTEMA: Pensão ; INDEXACAO: [Súmula, Guanabara, Custeio, Bombeiro militar, Polícia Militar, Pensão militar]",
  "PARAGRAFOLC": null,
  "REFERENCIALEGAL": null,
  "PUBLICACAOAPRESENTACAO": null,
  "PARADIGMATICO": "SUMULA"
}
```
### **Queries (`query.csv`)**

The `query.csv` file contains **150 standardized queries**, divided into three groups based on their origin and structure:

1. **G1 – Real User Queries**: 50 queries extracted from TCU search logs, representing actual user interactions.
2. **G2 – Synthetic Keyword-Based Queries**: 50 queries generated from the most accessed documents, keeping only essential terms.
3. **G3 – Synthetic Full-Sentence Queries**: 50 queries formulated as natural language questions derived from document summaries.

Each row in the file contains the following fields:

| Field   | Description |
|---------|------------|
| `ID`    | Unique query identifier |
| `TEXT`  | Query text (e.g., "técnica e preço") |
| `SOURCE` | Query origin (e.g., "search log", "synthetic") |

#### Example query entry
```json
{
    "ID": 1,
    "TEXT": "técnica e preço",
    "SOURCE": "search log"
}
```

### **Relevance Judgments (qrel.csv)**

The `qrel.csv` file contains relevance assessments linking queries to documents. These judgments were generated using a reranking pipeline followed by manual review.

Each row in the file contains the following fields:

| Field     | Description                                              |
|-----------|----------------------------------------------------------|
| **QUERY_ID** | Query identifier                                        |
| **DOC_ID**   | Document identifier                                     |
| **SCORE**    | Relevance score (0 = irrelevant, 3 = highly relevant)    |
| **ENGINE**   | Retrieval technique used (BM25, STS, Reranker, LLM)     |
| **RANK**     | Position of the document in the ranked retrieval list   |

#### Example relevance entry:

```json
{
    "QUERY_ID": 1,
    "DOC_ID": 21064,
    "SCORE": 3,
    "ENGINE": "(BM25|STS)+Reranker+LLM",
    "RANK": 1
}
```

## Citation
If you use the JurisTCU dataset in your research, please cite the following paper:

```bibtex
@article{juristcu2025,
  author = {Leandro Carísio Fernandes, Leandro dos Santos Ribeiro, Marcos Vinícius Borela de Castro, Leonardo Augusto da Silva Pacheco, Edans Flávius de Oliveira Sandes},
  title = {JurisTCU: A Brazilian Portuguese Information Retrieval Dataset with Query Relevance Judgments},
  journal = {To be published},
  year = {2025}
}
```

## License
JurisTCU is provided under the XXX license.

## Contact

- **Leandro Carísio Fernandes**  
  Câmara dos Deputados, Brasília, Brazil  
  Email: [carisio@gmail.com](mailto:carisio@gmail.com)

- **Leandro dos Santos Ribeiro**  
  Tribunal de Contas da União (TCU), Brasília, Brazil  
  Email: [leandro.santos.r@gmail.com](mailto:leandro.santos.r@gmail.co)

- **Marcos Vinícius Borela de Castro**  
  Tribunal de Contas da União (TCU), Brasília, Brazil  
  Email: [borela@tcu.gov.br](mailto:borela@tcu.gov.br)

- **Leonardo Augusto da Silva Pacheco**  
  Tribunal de Contas da União (TCU), Brasília, Brazil  
  Email: [leonardo3108@gmail.com](mailto:leonardo3108@gmail.com)

- **Edans Flávius de Oliveira Sandes**  
  Tribunal de Contas da União (TCU), Brasília, Brazil  
  Email: [edansfs@tcu.gov.br](mailto:edansfs@tcu.gov.br)


## Acknowledgments
We would like to thank the Brazilian Federal Court of Accounts (TCU) for providing the documents and supporting this research.