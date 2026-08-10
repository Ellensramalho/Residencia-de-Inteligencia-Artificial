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
    "retrieval_augmented_generation.md"
    "scaling_laws_llm.md"
]

PASTA_SAIDA = Path("chunks")

PASTA_SAIDA.mkdir(
    exist_ok=True
)

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)
if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY não encontrada. "
        "Configure a variável de ambiente."
    )
    
OPENROUTER_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)

MODELO = "openai/gpt-4o-mini"

def consultar_openrouter(prompt):

    headers = {
        "Authorization": (
            f"Bearer {OPENROUTER_API_KEY}"
        ),
        "Content-Type": "application/json"
    }

    payload = {

        "model": MODELO,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Você é um especialista em "
                    "chunking e sistemas RAG. "
                    "Avalie a qualidade dos chunks "
                    "de forma objetiva."
                )
            },

            {
                "role": "user",
                "content": prompt
            }

        ],

        "temperature": 0,

        "response_format": {
            "type": "json_object"
        }
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        resultado = response.json()

        conteudo = (
            resultado["choices"][0]
            ["message"]["content"]
        )

        return json.loads(conteudo)

    except Exception as erro:

        print(
            f"Erro ao consultar OpenRouter: {erro}"
        )

        return None

def avaliar_chunks_openrouter(
    chunks,
    nome_arquivo,
    numero_grupo
):

    amostra = chunks[:3]
    textos = []

    for i, chunk in enumerate(amostra):

        if isinstance(chunk, dict):

            texto = chunk.get(
                "text",
                ""
            )

        else:

            texto = str(chunk)

        textos.append(
            f"CHUNK {i + 1}:\n{texto}"
        )


    chunks_texto = "\n\n".join(
        textos
    )


    prompt = f"""
Estamos avaliando uma estratégia de chunking
para um sistema RAG.

Arquivo:
{nome_arquivo}

Grupo:
{numero_grupo}

Foram selecionados até 3 chunks como amostra.

Avalie essa estratégia considerando:

1. contexto:
O chunk mantém contexto suficiente para
ser compreendido isoladamente?

2. completude:
O chunk apresenta uma unidade de informação
relativamente completa?

3. qualidade:
Dê uma nota de 0 a 10 para a qualidade
dos chunks para uso em RAG.

4. justificativa:
Explique brevemente a avaliação.

Retorne SOMENTE um JSON válido no seguinte formato:

{{
    "contexto": "bom",
    "completude": "boa",
    "nota": 8,
    "justificativa": "..."
}}

Chunks:

{chunks_texto}
"""
    return consultar_openrouter(
        prompt
    )

def salvar_chunks(
    nome_arquivo,
    grupo,
    chunks
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

        if hasattr(
            chunk,
            "page_content"
        ):

            dados.append({

                "chunk_id": i + 1,

                "text":
                    chunk.page_content,

                "metadata":
                    chunk.metadata

            })

        elif isinstance(
            chunk,
            dict
        ):

            dados.append({

                "chunk_id": i + 1,
                "text":
                    chunk.get(
                        "text",
                        ""
                    ),

                "metadata":
                    chunk.get(
                        "metadata",
                        {}
                    )
            })
            
        else:
            dados.append({
                "chunk_id": i + 1,
                "text":
                    str(chunk),

                "metadata": {}

            })

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

    print(
        f"Grupo {grupo}: "
        f"{len(dados)} chunks → "
        f"{caminho_saida}"
    )

def grupo_1(texto):

    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=200,
        chunk_overlap=0
    )

    return splitter.split_text(
        texto
    )

def grupo_2(texto):

    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=500,
        chunk_overlap=0
    )

    return splitter.split_text(
        texto
    )

def grupo_3(texto):

    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=1000,
        chunk_overlap=0
    )

    return splitter.split_text(
        texto
    )

def grupo_4(texto):

    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=2000,
        chunk_overlap=0
    )

    return splitter.split_text(
        texto
    )

def grupo_5(texto):
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_text(
        texto
    )

def grupo_6(texto):
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=500,
        chunk_overlap=200
    )

    return splitter.split_text(
        texto
    )

def grupo_7(texto):

    paragrafos = re.split(
        r"\n\s*\n",
        texto
    )

    chunks = [
        paragrafo.strip()
        for paragrafo in paragrafos
        if paragrafo.strip()

    ]
    return chunks

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
    return splitter.split_text(
        texto
    )
    
def grupo_10(texto):
    headers_to_split_on = [
        ("#", "Título"),
        ("##", "Seção"),
        ("###", "Subseção"),
        ("####", "Subsubseção")

    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=
            headers_to_split_on,
        strip_headers=False

    )
    return splitter.split_text(
        texto
    )

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

nomes_estrategias = {
    1:
        "Caracteres - 200",
    2:
        "Caracteres - 500",
    3:
        "Caracteres - 1000",
    4:
        "Caracteres - 2000",
    5:
        "Caracteres - 500 + overlap 50",
    6:
        "Caracteres - 500 + overlap 200",
    7:
        "Parágrafos",
    8:
        "3 sentenças",
    9:
        "Recursive Character",
    10:
        "Markdown Headers"

}

avaliacoes = []

for nome_arquivo in arquivos:
    caminho_arquivo = (
        ARQUIVO / nome_arquivo
    )

    print("\n")
    print("=" * 70)
    
    print(
        f"PROCESSANDO: {nome_arquivo}"
    )

    print("=" * 70)

    if not caminho_arquivo.exists():
        print(
            f"ERRO: arquivo não encontrado: "
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
        f"Tamanho do documento: "
        f"{len(texto)} caracteres"
    )

    print("-" * 70)

    for numero_grupo, estrategia in (
        estrategias.items()
    ):
        print(
            f"\nProcessando grupo "
            f"{numero_grupo}..."
        )

        chunks = estrategia(
            texto
        )

        salvar_chunks(
            nome_arquivo,
            numero_grupo,
            chunks

        )
        print(
            "🤖 Avaliando com OpenRouter..."
        )
        
        avaliacao = (
            avaliar_chunks_openrouter(
                chunks,
                nome_arquivo,
                numero_grupo

            )
        )

        if avaliacao:
            nota = avaliacao.get(
                "nota",
                0
            )

            contexto = avaliacao.get(
                "contexto",
                ""
            )

            completude = avaliacao.get(
                "completude",
                ""
            )

            justificativa = (
                avaliacao.get(
                    "justificativa",
                    ""
                )
            )

        else:

            nota = 0
            contexto = (
                "Não avaliado"
            )
            completude = (
                "Não avaliado"
            )
        
            justificativa = (
                "Não foi possível "
                "consultar o OpenRouter."
            )

        tamanhos = [
            len(
                chunk.page_content
                
                if hasattr(
                    chunk,
                    "page_content"
                )
                else str(chunk)

            )
            for chunk in chunks

        ]

        avaliacoes.append({

            "Arquivo":
                nome_arquivo,

            "Grupo":
                numero_grupo,

            "Estratégia":
                nomes_estrategias[
                    numero_grupo
                ],

            "Total de chunks":
                len(chunks),

            "Tamanho médio":
                round(
                    sum(tamanhos)
                    / len(tamanhos),
                    2
                )
                if tamanhos
                else 0,

            "Menor":
                min(tamanhos)
                if tamanhos
                else 0,

            "Maior":
                max(tamanhos)
                if tamanhos
                else 0,

            "Contexto":
                contexto,

            "Completude":
                completude,

            "Nota IA":
                nota,

            "Justificativa":
                justificativa

        })


df = pd.DataFrame(
    avaliacoes
)

print("\n")

print("=" * 70)

print(
    "CHUNKING FINALIZADO!"
)
print("=" * 70)

print("\n")

print(df)

caminho_csv = (
    PASTA_SAIDA
    / "tabela_resumo.csv"
)

df.to_csv(
    caminho_csv,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"\n CSV salvo em:"
    f"\n{caminho_csv.resolve()}"
)

df_tabela = df[
    [
        "Arquivo",
        "Grupo",
        "Estratégia",
        "Total de chunks",
        "Tamanho médio",
        "Nota IA"
    ]
]

fig, ax = plt.subplots(
    figsize=(20, 10)
)

ax.axis("off")

table = ax.table(
    cellText=df_tabela.values,
    colLabels=df_tabela.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(
    False
)

table.set_fontsize(
    8
)

table.scale(
    1,
    2
)

plt.title(
    "Resumo das Estratégias de Chunking",
    fontsize=16,
    pad=20
)

caminho_imagem = (
    PASTA_SAIDA
    / "tabela_resumo.png"
)


plt.savefig(
    caminho_imagem,
    dpi=200,
    bbox_inches="tight"

)

plt.close()
print(
    f"\n Tabela salva em:"
    f"\n{caminho_imagem.resolve()}"
)

if not df.empty:

    melhor = df.loc[
        df["Nota IA"].idxmax()
    ]

    print("\n")
    print("=" * 70)
    print(
        " MELHOR ESTRATÉGIA SEGUNDO A IA"
    )

    print("=" * 70)

    print(
        f"Arquivo: "
        f"{melhor['Arquivo']}"
    )

    print(
        f"Grupo: "
        f"{melhor['Grupo']}"
    )

    print(
        f"Estratégia: "
        f"{melhor['Estratégia']}"
    )

    print(
        f"Nota: "
        f"{melhor['Nota IA']}/10"
    )

    print(
        f"Justificativa: "
        f"{melhor['Justificativa']}"
    )