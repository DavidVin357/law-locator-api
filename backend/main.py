import openai
import os
import pinecone

from dotenv import load_dotenv
from laws import get_paragraphs


load_dotenv()

emb_model_name = os.getenv("EMBEDDING_MODEL_NAME")

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.init(
    api_key=os.getenv("PINECONE_KEY"),
    environment=os.getenv("PINECONE_ENV"),  # find next to API key in console
)


def process_laws(laws: list[tuple]):
    total_embeds = 0
    if "openai" not in pinecone.list_indexes():
        pinecone.create_index("openai", dimension=1536)
    index = pinecone.Index("openai")

    for law_id, law_text in laws:
        paragraphs, paragraph_ids = get_paragraphs(law_text)

        response = openai.Embedding.create(input=paragraphs, model=emb_model_name)

        embeds = [
            (
                f"{law_id}|{paragraph_ids[i]}",
                record["embedding"],
                {"text": paragraphs[i]},
            )
            for i, record in enumerate(response["data"])
        ]

        print("embeds done: ", len(embeds))
        total_embeds += len(embeds)
        for i in range(0, len(embeds), 100):
            index.upsert(embeds[i : i + 100])

        print("upsert done!")

    print(total_embeds)


# laws = get_laws()
# process_laws(laws)
