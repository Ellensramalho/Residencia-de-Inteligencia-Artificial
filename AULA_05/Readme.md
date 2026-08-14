1. Qual campo incluiria se precisasse citar a fonte?

O campo 'fonte' identifica o arquivo de origem. O campo
'autor' também pode ser utilizado para informar quem é
o responsável pelo documento.

2. Por que chunk_index é útil?

O campo 'chunk_index' indica a posição do chunk dentro
do documento. Se o trecho recuperado estiver cortado
no meio de uma explicação, ele permite localizar os
chunks anterior e posterior para recuperar o contexto.
""") 

# 📚 LangChain Documents & Schema de Metadados para RAG

Este repositório contém a implementação prática da migração de chunks estruturados manualmente para o padrão nativo `Document` do ecossistema **LangChain**, com ênfase no design de schemas de metadados ricos para sistemas de **Retrieval-Augmented Generation (RAG)**.

---

## 🚀 Sobre o Projeto

Em pipelines de RAG profissionais, a qualidade da recuperação de contexto depende tanto da busca vetorial quanto da capacidade de rastrear, filtrar e enriquecer os dados antes e depois da indexação.

Este projeto aborda:
- A anatomia do objeto `Document` do LangChain (`page_content` vs. `metadata`).
- Comportamento de tipos em metadados e restrições práticas em Vector Stores.
- Engenharia e design de **Schemas de Metadados** para citação de fontes e *Sentence/Chunk Windowing*.
- Extração automatizada de metadados em arquivos Markdown locais.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.10+**
- **LangChain Core** (`langchain-core`)
- **Pandas** (Estruturação e exibição do schema)
- **Pathlib & Regular Expressions (re)** (Ingestão e parsing de arquivos)

---

## 📦 Estrutura do Schema de Metadados

O schema foi projetado para garantir máxima rastreabilidade sem inflar desnecessariamente o índice vetorial:

| Campo | Tipo | Descrição | Utilidade no RAG |
|---|---|---|---|
| `fonte` | `string` | Nome do arquivo `.md` de origem | Citação direta e auditoria de resposta |
| `documento_id` | `string` | Identificador único do documento | Agrupamento de chunks do mesmo arquivo |
| `chunk_index` | `integer` | Posição sequencial do chunk | Recuperação de vizinhança (*window retrieval*) |
| `estrategia` | `string` | Técnica de corte usada | Rastreabilidade do pipeline de ingestão |
| `chunk_size` | `integer` | Janela configurada | Controle de granularidade |
| `chunk_overlap` | `integer` | Sobreposição configurada | Diagnóstico de redundância |
| `n_caracteres` | `integer` | Contagem real de caracteres | Validação de densidade textual |
| `titulo` | `string` | Título extraído do cabeçalho Markdown | Injeção de contexto temático no prompt |
| `autor` | `string` | Autor/origem do material | Atribuição de autoridade e citação |
| `tipo_conteudo` | `string` | Ex: `teoria`, `exemplo`, `exercicio` | Metadata Filtering pré-busca vetorial |
| `data_processamento` | `string` | Data de ingestão (ISO 8601) | Invalidação de cache e controle de versão |
