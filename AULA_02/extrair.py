import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


def extrair_metadados(markdown):

    resposta = client.chat.completions.create(
        model="openai/gpt-4.1-mini",

        messages=[
            {
                "role": "system",
                "content": """
                Você é um extrator de metadados de artigos científicos.
                Extraia apenas as informações solicitadas.
                Caso alguma informação não exista, use null.
                """
            },
            {
                "role": "user",
                "content": markdown
            }
        ],

        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "metadados_documento",
                "schema": {
                    "type": "object",
                    "properties": {
                        "titulo": {
                            "type": "string"
                        },
                        "autores": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "ano": {
                            "type": "integer"
                        }
                    },
                    "required": [
                        "titulo",
                        "autores",
                        "ano"
                    ],
                    "additionalProperties": False
                }
            }
        }
    )


    return json.loads(
        resposta.choices[0].message.content
    )


with open(
    "bioetica_e_ia.md",
    "r",
    encoding="utf-8"
) as arquivo:

    conteudo = arquivo.read()


metadados = extrair_metadados(conteudo)


print(
    json.dumps(
        metadados,
        indent=2,
        ensure_ascii=False
    )
)