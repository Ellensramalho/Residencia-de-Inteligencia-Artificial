import os
import re
import requests
import numpy as np

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY não encontrada no arquivo .env"
    )

URL = "https://openrouter.ai/api/v1/embeddings"
MODEL = "openai/text-embedding-3-small"

print("OpenRouter configurado!")


PASTA = "markdowns"

arquivos = [
    "bioetica_e_ia.md",
    "escrita_academica_ia.md",
    "twitter_algoritmo.md"
]

def gerar_embeddings(textos):

    if not textos:
        return []

    response = requests.post(

        URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Busca Semantica"
        },

        json={
            "model": MODEL,
            "input": textos
        },
        timeout=120
    )
    
    if response.status_code != 200:
        raise Exception(
            f"Erro {response.status_code}: "
            f"{response.text}"
        )

    dados = response.json()

    embeddings = sorted(
        dados["data"],
        key=lambda x: x["index"]
    )


    return [
        item["embedding"]
        for item in embeddings
    ]

def gerar_embeddings_trechos(
    trechos,
    tamanho_lote=50
):

    resultados = []
    
    total = len(trechos)
    print(
        f"\n Gerando embeddings para {total} trechos..."
    )

    for inicio in range(
        0,
        total,
        tamanho_lote
    ):

        fim = min(
            inicio + tamanho_lote,
            total
        )

        lote = trechos[inicio:fim]

        textos = [
            trecho["texto"]
            for trecho in lote
        ]

        print(
            f"Processando trechos "
            f"{inicio + 1} até {fim} "
            f"de {total}..."
        )
        
        try:
            embeddings = gerar_embeddings(
                textos
            )

            for trecho, embedding in zip(
                lote,
                embeddings
            ):
                resultados.append({

                    "arquivo": trecho["arquivo"],

                    "tipo": trecho["tipo"],

                    "texto": trecho["texto"],

                    "embedding": embedding
                })

        except Exception as erro:
            print(
                f"\n Erro no lote "
                f"{inicio + 1}-{fim}:"
            )
            print(erro)

    print(
        f"\n {len(resultados)} embeddings gerados."
    )
    return resultados

def similaridade_cosseno(
    vetor1,
    vetor2
):
    vetor1 = np.array(vetor1)
    vetor2 = np.array(vetor2)

    produto = np.dot(
        vetor1,
        vetor2
    )

    norma1 = np.linalg.norm(
        vetor1
    )

    norma2 = np.linalg.norm(
        vetor2
    )
    
    if norma1 == 0 or norma2 == 0:
        return 0

    return produto / (
        norma1 * norma2
    )

documentos = []

for nome_arquivo in arquivos:
    caminho = os.path.join(
        PASTA,
        nome_arquivo
    )
    
    if not os.path.exists(caminho):
        print(
            f" Arquivo não encontrado: "
            f"{caminho}"
        )
        continue

    with open(
        caminho,
        "r",
        encoding="utf-8"
    ) as arquivo:
        conteudo = arquivo.read()

    documentos.append({
        "arquivo": nome_arquivo,
        "conteudo": conteudo
    })

print("\ Arquivos encontrados:")

for documento in documentos:
    print(
        f"- {documento['arquivo']}"
    )

if not documentos:
    raise ValueError(
        "Nenhum arquivo Markdown foi encontrado."
    )

def dividir_linhas(documento):
    linhas = documento[
        "conteudo"
    ].splitlines()

    trechos = []

    for linha in linhas:
        linha = linha.strip()

        if linha:
            trechos.append({

                "arquivo": documento[
                    "arquivo"
                ],
                "tipo": "linha",

                "texto": linha
            })

    return trechos

def dividir_paragrafos(documento):

    paragrafos = re.split(

        r"\n\s*\n",

        documento["conteudo"]
    )
    
    trechos = []
    
    for paragrafo in paragrafos:
        paragrafo = paragrafo.strip()
        
        if paragrafo:
            trechos.append({
                "arquivo": documento[
                    "arquivo"
                ],
                "tipo": "parágrafo",
                "texto": paragrafo

            })
    return trechos

def dividir_capitulos(documento):
    partes = re.split(
        r"(?m)(?=^#{1,3}\s+)",
        documento["conteudo"]
    )
    trechos = []

    for parte in partes:
        parte = parte.strip()
        if parte:
            trechos.append({
                "arquivo": documento[
                    "arquivo"
                ],
                "tipo": "capítulo",
                "texto": parte
            })

    return trechos
def busca_semantica(
    query,
    trechos,
    top_k=3
):
    print("\n")
    print("=" * 50)
    print("BUSCA SEMÂNTICA")
    print("=" * 50)
    print("\nQuery:")
    print(query)
    
    embedding_query = gerar_embeddings(
        [query]
    )[0]
    resultados = []
    for trecho in trechos:
        score = similaridade_cosseno(
            embedding_query,
            trecho["embedding"]
        )

        resultados.append({
            "arquivo": trecho[
                "arquivo"
            ],
            "tipo": trecho[
                "tipo"
            ],
            "texto": trecho[
                "texto"
            ],
            "score": score
        })
        
    resultados.sort(
        key=lambda x: x[
            "score"
        ],
        reverse=True
    )
    print("\n TOP 3 RESULTADOS")
    print("=" * 50)
    
    for i, resultado in enumerate(
        resultados[:top_k],
        start=1
    ):
        print(
            f"\n Resultado {i}"
        )
        
        print(
            f" Arquivo: "
            f"{resultado['arquivo']}"
        )
        
        print(
            f" Tipo: "
            f"{resultado['tipo']}"
        )
        
        print(
            f" Score: "
            f"{resultado['score']:.4f}"
        )
        print("\nTexto:")

        print(
            resultado["texto"]
        )
        print(
            "\n" + "-" * 70
        )

    return resultados[:top_k]
  
print("\n")
print("=" * 50)
print(" BUSCA POR LINHA")
print("=" * 50)

trechos_linhas = []

for documento in documentos:
    trechos_linhas.extend(
        dividir_linhas(documento)
    )
    
print(
    f"\n Total de linhas: "
    f"{len(trechos_linhas)}"
)

embeddings_linhas = (
    gerar_embeddings_trechos(
        trechos_linhas
    )
)

query = input(
    "\n Digite sua pergunta: "
)

resultados_linha = busca_semantica(

    query,
    embeddings_linhas,
    top_k=3
)

print("\n")
print("=" * 50)
print(" BUSCA POR PARÁGRAFO")
print("=" * 50)

trechos_paragrafos = []

for documento in documentos:
    trechos_paragrafos.extend(
        dividir_paragrafos(documento)
    )

print(
    f"\n Total de parágrafos: "
    f"{len(trechos_paragrafos)}"
)

embeddings_paragrafos = (
    gerar_embeddings_trechos(
        trechos_paragrafos
    )
)

resultados_paragrafo = busca_semantica(

    query,
    embeddings_paragrafos,
    top_k=3

)

print("\n")
print("=" * 50)
print("BUSCA POR CAPÍTULO")
print("=" * 50)

trechos_capitulos = []

for documento in documentos:
    trechos_capitulos.extend(
        dividir_capitulos(documento)

    )

print(
    f"\n Total de capítulos: "
    f"{len(trechos_capitulos)}"
)

embeddings_capitulos = (
    gerar_embeddings_trechos(
        trechos_capitulos
    )
)

resultados_capitulo = busca_semantica(
    query,
    embeddings_capitulos,
    top_k=3

)

print("\n")
print("=" * 50)

print(
    " BUSCA SEMÂNTICA CONCLUÍDA!"
)

print("=" * 50)