import requests

from xml.dom.minidom import parseString
import tiktoken

import os
from dotenv import load_dotenv

load_dotenv()

# Weapons act
# Tobacco act
# Traffic Act
# Road Transport Act
# Food act

#  ids = [
#             "520092023002",
#             "503052023002",
#             "516022016004",
#             "530062023008",
#             "506102023009",
#         ]

with open("law-ids.txt", "r") as file:
    ids = [id.strip() for id in file]


xml_url = "https://www.riigiteataja.ee/en/tolge/xml/"

emb_model_name = os.getenv("EMBEDDING_MODEL_NAME")


def get_laws():
    laws = []
    for id in ids:
        law_url = xml_url + id
        response = requests.get(law_url)

        if response.status_code == 200:
            laws.append((id, response.text))

    return laws


def get_law_structure(law: str):
    lawNode = parseString(law)

    law_title = (
        lawNode.getElementsByTagName("nimi")[0]
        .getElementsByTagName("pealkiri")[0]
        .firstChild.nodeValue
    )

    tokenizer = tiktoken.encoding_for_model(emb_model_name)
    paragraphs = []
    ids = []
    for p in lawNode.getElementsByTagName("loige"):
        paragraph_id = p.getAttribute("id")
        textNodes = p.getElementsByTagName("tavatekst")
        paragraph_text = " ".join([node.firstChild.nodeValue for node in textNodes])

        if paragraph_text.strip() and len(tokenizer.encode(paragraph_text)) <= 8192:
            paragraphs.append(paragraph_text)
            ids.append(paragraph_id)
    return law_title, paragraphs, ids
