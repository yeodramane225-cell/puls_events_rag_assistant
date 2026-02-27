"""
Orchestration RAG avec LangChain pour Puls-Events.

- Embeddings Mistral
- Index FAISS existant
- Filtrage par année
- RAG orchestré via LCEL
"""

import json
import re
import numpy as np
import faiss
import pandas as pd

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from mistralai import Mistral

MISTRAL_API_KEY = "koYSEltxtO2OhW0twIdOJTzbZDxYUEzt"
EMBED_MODEL = "mistral-embed"
LLM_MODEL = "mistral-large-latest"

# ---------------------------------------------------------
# 1. Embeddings Mistral
# ---------------------------------------------------------
client = Mistral(api_key=MISTRAL_API_KEY)

def embed_mistral(text: str):
    emb = client.embeddings.create(
        model=EMBED_MODEL,
        inputs=[text]   # correct pour mistralai 1.12.4
    )
    return np.array(emb.data[0].embedding, dtype="float32")


# ---------------------------------------------------------
# 2. Charger FAISS + metadata
# ---------------------------------------------------------
index = faiss.read_index("data/vectorstore/faiss_index.bin")

with open("data/vectorstore/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

df_chunks = pd.DataFrame(metadata)


# ---------------------------------------------------------
# 3. Filtrage par année
# ---------------------------------------------------------
def filter_by_year(question: str):
    match = re.search(r"\b(19\d{2}|20\d{2})\b", question)
    return match.group(0) if match else None


def retrieve_chunks(question: str, k=5):
    year = filter_by_year(question)

    q_emb = embed_mistral(question).reshape(1, -1)

    if year:
        filtered_df = df_chunks[df_chunks["date_start"].str.contains(year, na=False)]

        if len(filtered_df) == 0:
            return "Aucun événement trouvé pour l'année " + year

        sub_index = faiss.IndexFlatL2(index.d)

        emb_list = []
        for original_idx in filtered_df.index:
            emb = index.reconstruct(original_idx)
            emb_list.append(emb)

        sub_index.add(np.vstack(emb_list))

        distances, indices = sub_index.search(q_emb, min(k, len(filtered_df)))

        return "\n".join(filtered_df.iloc[i]["chunk"] for i in indices[0])

    distances, indices = index.search(q_emb, k)
    return "\n".join(df_chunks.iloc[i]["chunk"] for i in indices[0])


# ---------------------------------------------------------
# 4. Prompt LangChain
# ---------------------------------------------------------
prompt = PromptTemplate.from_template("""
Tu es un assistant spécialisé dans les événements publics à Paris.

Voici des informations pertinentes extraites de la base OpenAgenda :
{context}

Question : {question}

Réponds de manière claire, concise et utile.
""")


# ---------------------------------------------------------
# 5. LLM Mistral via LangChain
# ---------------------------------------------------------
def call_mistral(prompt_text):
    # LangChain envoie un StringPromptValue → conversion obligatoire
    if hasattr(prompt_text, "to_string"):
        prompt_text = prompt_text.to_string()
    else:
        prompt_text = str(prompt_text)

    resp = client.chat.complete(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------
# 6. Pipeline LCEL
# ---------------------------------------------------------
rag_chain = (
    RunnableParallel(
        context=RunnableLambda(lambda x: retrieve_chunks(x["question"])),
        question=RunnablePassthrough()
    )
    | prompt
    | RunnableLambda(call_mistral)
    | StrOutputParser()
)


# ---------------------------------------------------------
# 7. Fonction publique
# ---------------------------------------------------------
def rag_langchain(question: str):
    return rag_chain.invoke({"question": question})
