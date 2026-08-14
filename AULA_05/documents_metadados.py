import json
import random
import re
from datetime import date
from pathlib import Path
import pandas as pd
from langchain_core.documents import Document

documentos = [
    Document(
        page_content="Embeddings são representações vetoriais densas de texto que preservam relações semânticas.",
        metadata={
            "fonte": "attention_is_all_you_need.md",
            "pagina": 1,
            "tipo": "teoria",
            "tema": "embeddings",
            "autor": "Vaswani et al."
        }
    ),
    Document(
        page_content="Chunking divide documentos grandes em partes menores para caber na janela de contexto dos modelos.",
        metadata={
            "fonte": "bert_pretraining.md",
            "pagina": 2,
            "tipo": "teoria",
            "tema": "chunking",
            "autor": "Devlin et al."
        }
    ),
    Document(
        page_content="RAG recupera informações relevantes em uma base de conhecimento antes de gerar a resposta final.",
        metadata={
            "fonte": "rag_survey.md",
            "pagina": 3,
            "tipo": "teoria",
            "tema": "RAG",
            "autor": "Lewis et al."
        }
    ),
    Document(
        page_content="Tokenização transforma cadeias de caracteres em sequências de tokens que o modelo consegue processar.",
        metadata={
            "fonte": "tokenizers_intro.md",
            "pagina": 4,
            "tipo": "teoria",
            "tema": "tokenização",
            "autor": "HuggingFace Team"
        }
    ),
    Document(
        page_content="Podemos calcular a similaridade de cosseno entre dois embeddings para medir a afinidade dos textos.",
        metadata={
            "fonte": "vector_search_guide.md",
            "pagina": 5,
            "tipo": "exemplo",
            "tema": "embeddings",
            "autor": "Pinecone Team"
        }
    )
]

for i, doc in enumerate(documentos, start=1):
    print(f"\n--- Documento {i} ---")
    print(f"page_content: {doc.page_content}")
    print(f"metadata:     {doc.metadata}")

print(f"\nTotal de documentos criados: {len(documentos)}")

print("\n" + "-" * 40)
print("TESTES DE COMPORTAMENTO DO DOCUMENT")
print("-" * 40)

doc_aninhado = Document(
    page_content="Documento com listas e dicionários aninhados.",
    metadata={
        "fonte": "teste.md",
        "tags": ["IA", "RAG", "Embeddings"],
        "info": {"revisado": True, "versao": 1.2}
    }
)
print("Documento com metadados complexos:", doc_aninhado)

doc_vazio = Document(page_content="Documento sem metadados explícitos.")
print("Metadados padrão quando omitidos:", doc_vazio.metadata)

print("\n" + "=" * 60)
print("EXERCÍCIO 2 - SCHEMA DE METADADOS")
print("=" * 60)

schema = [
    {"campo": "fonte", "tipo": "string", "descricao": "Nome do arquivo .md de origem"},
    {"campo": "documento_id", "tipo": "string", "descricao": "Identificador do documento"},
    {"campo": "chunk_index", "tipo": "integer", "descricao": "Posição sequencial do chunk no documento"},
    {"campo": "estrategia", "tipo": "string", "descricao": "Estratégia de chunking utilizada"},
    {"campo": "chunk_size", "tipo": "integer", "descricao": "Configuração de tamanho da janela"},
    {"campo": "chunk_overlap", "tipo": "integer", "descricao": "Configuração de sobreposição"},
    {"campo": "n_caracteres", "tipo": "integer", "descricao": "Tamanho real do texto em caracteres"},
    {"campo": "titulo", "tipo": "string", "descricao": "Título extraído do documento"},
    {"campo": "autor", "tipo": "string", "descricao": "Autor ou fonte original do material"},
    {"campo": "tipo_conteudo", "tipo": "string", "descricao": "Classificação do conteúdo (teoria, exemplo, etc.)"},
    {"campo": "data_processamento", "tipo": "string", "descricao": "Data da indexação/processamento"}
]

df_schema = pd.DataFrame(schema)
print("\nTabela do Schema Definido:\n")
print(df_schema.to_string(index=False))

pasta_markdowns = Path("markdowns")
arquivos = list(pasta_markdowns.glob("*.md"))

if not arquivos:
    nome_fonte = "exemplo_aula04.md"
    documento_id = "exemplo_aula04"
    titulo = "Introdução a Arquiteturas de RAG"
    autor = "Equipe do Curso"
    texto_chunk = "O Retrieval-Augmented Generation (RAG) combina busca vetorial com modelos generativos..."
else:
    arquivo_escolhido = random.choice(arquivos)
    nome_fonte = arquivo_escolhido.name
    documento_id = arquivo_escolhido.stem
    texto = arquivo_escolhido.read_text(encoding="utf-8")

    match_titulo = re.search(r"^#\s+(.+)$", texto, re.MULTILINE)
    titulo = match_titulo.group(1).strip() if match_titulo else arquivo_escolhido.stem

    match_autor = re.search(r"(?im)^(?:autor|author)\s*:\s*(.+)$", texto)
    autor = match_autor.group(1).strip() if match_autor else "Não informado"

    chunk_size = 500
    texto_chunk = texto[:chunk_size]

chunk_exemplo = {
    "text": texto_chunk,
    "metadata": {
        "fonte": nome_fonte,
        "documento_id": documento_id,
        "chunk_index": 0,
        "estrategia": "recursive_character",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "n_caracteres": len(texto_chunk),
        "titulo": titulo,
        "autor": autor,
        "tipo_conteudo": "teoria",
        "data_processamento": str(date.today())
    }
}

print("\n" + "=" * 60)
print("EXEMPLO DE CHUNK PROCESSADO (JSON)")
print("=" * 60)
print(json.dumps(chunk_exemplo, indent=4, ensure_ascii=False))
