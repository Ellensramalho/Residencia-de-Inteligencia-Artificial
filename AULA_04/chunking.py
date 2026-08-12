import os
import re
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)

load_dotenv()

ARQUIVO = Path("markdowns")
PASTA_SAIDA = Path("chunks")

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True
)

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY não encontrada. "
        "Configure a variável no arquivo .env"
    )

OPENROUTER_EMBEDDING_URL = (
    "https://openrouter.ai/api/v1/embeddings"
)

MODELO_EMBEDDING = (
    "openai/text-embedding-3-small"
)

EMBEDDINGS_POR_GRUPO = 5

arquivos = [
    "bioetica_e_ia.md",
    "escrita_academica_ia.md",
    "twitter_algoritmo.md",
    "attention_is_all_you_need.md",
    "bert_pretraining.md",
    "gpt3_language_models.md",
    "gpt4_technical_report.md",
    "instruct_gpt.md",
    "llama_foundation_models.md",
    "lora_low_rank_adaptation.md",
    "retrieval_augmented_generation.md",
    "scaling_laws_llm.md"
]

def grupo_1(texto):
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=200,
        chunk_overlap=0
    )

    return splitter.split_text(texto)

def grupo_2(texto):
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=500,
        chunk_overlap=0
    )
    return splitter.split_text(texto)

def grupo_3(texto):
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=1000,
        chunk_overlap=0
    )

    return splitter.split_text(texto)

def grupo_4(texto):
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=2000,
        chunk_overlap=0
    )

    return splitter.split_text(texto)

def grupo_5(texto):
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_text(texto)

def grupo_6(texto):
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=500,
        chunk_overlap=200
    )

    return splitter.split_text(texto)

def grupo_7(texto):
    paragrafos = re.split(
        r"\n\s*\n",
        texto
    )

    return [
        paragrafo.strip()
        for paragrafo in paragrafos
        if paragrafo.strip()
    ]

def grupo_8(texto):

    sentencas = re.split(
        r"(?<=[.!?])\s+",
        texto
    )

    sentencas = [
        sentenca.strip()
        for sentenca in sentencas
        if sentenca.strip()
    ]

    chunks = []

    for i in range(
        0,
        len(sentencas),
        3
    ):

        grupo = sentencas[
            i:i + 3
        ]

        chunks.append(
            " ".join(grupo)
        )

    return chunks

def grupo_9(texto):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
            ""
        ]
    )

    return splitter.split_text(texto)

def grupo_10(texto):

    headers_to_split_on = [
        ("#", "Título"),
        ("##", "Seção"),
        ("###", "Subseção"),
        ("####", "Subsubseção")
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    return splitter.split_text(texto)

estrategias = {
    1: grupo_1,
    2: grupo_2,
    3: grupo_3,
    4: grupo_4,
    5: grupo_5,
    6: grupo_6,
    7: grupo_7,
    8: grupo_8,
    9: grupo_9,
    10: grupo_10

}


informacoes_testes = {

    1: {
        "estrategia": "Fixo",
        "configuracao": "200 caracteres, sem overlap",
        "variavel": "Tamanho extremo baixo"
    },

    2: {
        "estrategia": "Fixo",
        "configuracao": "500 caracteres, sem overlap",
        "variavel": "Tamanho"
    },

    3: {
        "estrategia": "Fixo",
        "configuracao": "1000 caracteres, sem overlap",
        "variavel": "Tamanho"
    },

    4: {
        "estrategia": "Fixo",
        "configuracao": "2000 caracteres, sem overlap",
        "variavel": "Tamanho extremo alto"
    },

    5: {
        "estrategia": "Fixo + overlap",
        "configuracao": "500 caracteres, overlap 50",
        "variavel": "Overlap leve"
    },

    6: {
        "estrategia": "Fixo + overlap",
        "configuracao": "500 caracteres, overlap 200",
        "variavel": "Overlap pesado"
    },

    7: {
        "estrategia": "Por parágrafo",
        "configuracao": "Separação por parágrafos",
        "variavel": "Estrutura natural"
    },

    8: {
        "estrategia": "Por sentença",
        "configuracao": "Sentenças agrupadas em 3",
        "variavel": "Estrutura natural"
    },

    9: {
        "estrategia": "Recursivo",
        "configuracao": "Separadores hierárquicos",
        "variavel": "Estratégia composta"
    },

    10: {
        "estrategia": "Markdown",
        "configuracao": "Separação por headings/seções",
        "variavel": "Estrutura semântica"
    }

}

def gerar_embedding(texto):
    headers = {
        "Authorization":
            f"Bearer {OPENROUTER_API_KEY}",

        "Content-Type":
            "application/json"
    }

    payload = {
        "model":
            MODELO_EMBEDDING,

        "input":
            [texto]
    }

    try:
        response = requests.post(
            OPENROUTER_EMBEDDING_URL,
            headers=headers,
            json=payload,
            timeout=120
        )

        response.raise_for_status()
        resultado = response.json()
        return resultado["data"][0]["embedding"]

    except Exception as erro:
        print(
            f"Erro ao gerar embedding: {erro}"
        )

        return None


def selecionar_chunks_para_embedding(chunks):
    quantidade = len(chunks)

    if quantidade == 0:
        return []

    quantidade_embeddings = min(
        EMBEDDINGS_POR_GRUPO,
        quantidade
    )

    if quantidade <= EMBEDDINGS_POR_GRUPO:

        return list(
            range(quantidade)
        )

    indices = []

    passo = (
        quantidade - 1
    ) / (
        quantidade_embeddings - 1
    )

    for i in range(
        quantidade_embeddings
    ):

        indice = round(
            i * passo
        )

        if indice not in indices:

            indices.append(
                indice
            )

    return indices

def obter_texto_chunk(chunk):

    if hasattr(
        chunk,
        "page_content"
    ):

        return chunk.page_content

    elif isinstance(
        chunk,
        dict
    ):

        return chunk.get(
            "text",
            ""
        )

    else:

        return str(chunk)


def salvar_chunks(
    nome_arquivo,
    grupo,
    chunks,
    embeddings
):

    nome_base = Path(
        nome_arquivo
    ).stem

    pasta_grupo = (
        PASTA_SAIDA
        / nome_base
        / f"grupo_{grupo}"
    )

    pasta_grupo.mkdir(
        parents=True,
        exist_ok=True
    )

    caminho_saida = (
        pasta_grupo
        / "chunks.json"
    )

    dados = []

    for i, chunk in enumerate(chunks):
        texto = obter_texto_chunk(
            chunk
        )

        if hasattr(
            chunk,
            "metadata"
        ):

            metadata = chunk.metadata

        elif isinstance(
            chunk,
            dict
        ):

            metadata = chunk.get(
                "metadata",
                {}
            )

        else:
            metadata = {}

        item = {
            "chunk_id":
                i + 1,

            "text":
                texto,

            "metadata":
                metadata
        }

        if i in embeddings:
            item["embedding"] = (
                embeddings[i]
            )

        dados.append(
            item
        )

    with open(
        caminho_saida,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dados,
            f,
            ensure_ascii=False,
            indent=2
        )

    quantidade_embeddings = sum(
        1
        for item in dados
        if "embedding" in item
    )

    print(
        f"Grupo {grupo}: "
        f"{len(dados)} chunks | "
        f"{quantidade_embeddings} embeddings"
    )


total_embeddings = 0

dados_tabela = []


for nome_arquivo in arquivos:

    caminho_arquivo = (
        ARQUIVO
        / nome_arquivo
    )

    print("\n")
    print("=" * 70)

    print(
        f"PROCESSANDO: {nome_arquivo}"
    )

    print("=" * 70)

    if not caminho_arquivo.exists():

        print(
            f"Arquivo não encontrado: "
            f"{caminho_arquivo}"
        )

        continue

    with open(
        caminho_arquivo,
        "r",
        encoding="utf-8"
    ) as f:

        texto = f.read()

    print(
        f"Tamanho: "
        f"{len(texto)} caracteres"
    )

    for numero_grupo, estrategia in (
        estrategias.items()
    ):

        print()

        print(
            f"Processando grupo {numero_grupo}: "
            f"{informacoes_testes[numero_grupo]['estrategia']}"
        )

        chunks = estrategia(
            texto
        )

        quantidade_chunks = len(
            chunks
        )

        print(
            f"Total de chunks: "
            f"{quantidade_chunks}"
        )

        dados_tabela.append({

            "Arquivo":
                nome_arquivo,

            "Teste":
                numero_grupo,

            "Estratégia":
                informacoes_testes[
                    numero_grupo
                ]["estrategia"],

            "Configuração":
                informacoes_testes[
                    numero_grupo
                ]["configuracao"],

            "Variável isolada":
                informacoes_testes[
                    numero_grupo
                ]["variavel"],

            "Quantidade de chunks":
                quantidade_chunks
        })

        indices_embeddings = (
            selecionar_chunks_para_embedding(
                chunks
            )
        )

        print(
            f"Chunks selecionados para "
            f"embedding: "
            f"{indices_embeddings}"
        )

        embeddings = {}

        for indice in indices_embeddings:

            texto_embedding = (
                obter_texto_chunk(
                    chunks[indice]
                )
            )

            print(
                f"Gerando embedding "
                f"{len(embeddings) + 1}/"
                f"{len(indices_embeddings)}..."
            )

            embedding = gerar_embedding(
                texto_embedding
            )

            if embedding:

                embeddings[indice] = (
                    embedding
                )

                total_embeddings += 1

                print(
                    "Embedding gerado."
                )

            else:

                print(
                    "Falha ao gerar embedding."
                )
                
        salvar_chunks(
            nome_arquivo,
            numero_grupo,
            chunks,
            embeddings
        )

df_detalhada = pd.DataFrame(
    dados_tabela
)

resultados_representativos = []

for teste in sorted(
    df_detalhada["Teste"].unique()
):

    dados_teste = df_detalhada[
        df_detalhada["Teste"] == teste
    ].copy()

    mediana = dados_teste[
        "Quantidade de chunks"
    ].median()

    dados_teste["distancia"] = (
        dados_teste[
            "Quantidade de chunks"
        ] - mediana
    ).abs()

    melhor = dados_teste.sort_values(
        [
            "distancia",
            "Quantidade de chunks"
        ]
    ).iloc[0]

    resultados_representativos.append({

        "Teste":
            melhor["Teste"],

        "Estratégia":
            melhor["Estratégia"],

        "Arquivo representativo":
            melhor["Arquivo"],

        "Configuração":
            melhor["Configuração"],

        "Quantidade de chunks":
            melhor["Quantidade de chunks"]

    })


df_tabela = pd.DataFrame(
    resultados_representativos
)

caminho_csv = (
    PASTA_SAIDA
    / "tabela_resultados_representativos.csv"
)

df_tabela.to_csv(
    caminho_csv,
    index=False,
    encoding="utf-8-sig"
)


print("\n")
print("=" * 70)

print(
    "RESULTADOS REPRESENTATIVOS"
)

print("=" * 70)
print()

print(
    df_tabela.to_string(
        index=False
    )
)

print()

print(
    f"Tabela CSV salva em:"
    f"\n{caminho_csv.resolve()}"
)

fig, ax = plt.subplots(
    figsize=(16, 6)
)

ax.axis("off")

tabela = ax.table(
    cellText=df_tabela.values,
    colLabels=df_tabela.columns,
    cellLoc="left",
    loc="center"
)

tabela.auto_set_font_size(
    False
)

tabela.set_fontsize(
    9
)

tabela.scale(
    1,
    2
)

plt.title(
    "Resultados Representativos das Estratégias de Chunking",
    fontsize=16,
    pad=20
)

caminho_imagem = (
    PASTA_SAIDA
    / "tabela_resultados_representativos.png"
)

plt.savefig(
    caminho_imagem,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    f"Tabela PNG salva em:"
    f"\n{caminho_imagem.resolve()}"
)

print("\n")
print("=" * 70)

print(
    "PROCESSAMENTO FINALIZADO!"
)

print("=" * 70)
print()

print(
    f"{len(arquivos)} arquivos processados."
)

print(
    f"Até {EMBEDDINGS_POR_GRUPO} embeddings "
    f"por grupo."
)

print(
    f"Total de embeddings gerados: "
    f"{total_embeddings}"
)

print(
    "Tabela final contém apenas "
    f"{len(df_tabela)} resultados representativos."
)

print(
    "O arquivo selecionado em cada estratégia é o mais próximo da mediana de chunks."
)