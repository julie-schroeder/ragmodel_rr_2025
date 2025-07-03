### Formål
# Scriptet indlæser json filerne, embedder dem og gemmer dem så i et vector_store format
# Kræver en API key

import json
import os

##Indlæser data fra json-databasen i forrige script
documents = []

with open("json_database/rag_chunks.jsonl", "r") as f:
    for line in f:
        item = json.loads(line) #loads fordi det er et jsonl format i stedet for json
        documents.append(item) #henter fra json


##konverterer fra json til document-objekter, som kan bruges i langchain

#LangChains Document-klasse er en struktur, der indeholder: 
# page_content (str): Det faktiske indhold, man vil give til modellen.
#metadata (dict): Ekstra info, fx kilde, titel, sidetal, etc.
from langchain.docstore.document import Document
docs = [Document(page_content=d["content"], metadata=d["metadata"]) for d in documents] #binder det sammen i et langchain dokument


#Opsætter til API
from dotenv import load_dotenv
load_dotenv() ##henter key
print("API key:", os.getenv("OPENAI_API_KEY"))

#opsætter embedding-model til openAI
from langchain_openai import OpenAIEmbeddings
import openai
embedding_model = OpenAIEmbeddings()

#Gemmer det embeddede materiale som vector store med FAISS. 
from langchain.vectorstores import FAISS 
db = FAISS.from_documents(docs, embedding_model)
db.save_local("vector_store")

##########################################################
##test af embedding-modellen
# Brugerspørgsmål / prompt
query = "Hvad har Rigsrevisionen konkluderet i beretningen om palliation?"
aar_tidligst = 2022
aar_senest = 2025
dokument_type = "beretning"

# Søg i FAISS efter de mest relevante chunks
##her kunne man lægge et filter ind med search_kwargs("filter"), men den filtrerer først efter embedding-modellen er kørt.

#manuelt filter før embedding-model og opretter ny vector-databasen
filtrede_doks = [doc for doc in docs if (
                     aar_tidligst <= doc.metadata.get("year", 0) <= aar_senest and
                     doc.metadata.get("dokumenttype") == dokument_type)] 
filtreret_db = FAISS.from_documents(filtrede_doks, embedding_model)

#embedding-modellen søger efter k matches
results = db.similarity_search(query, k=5)  # k = hvor mange top-resultater du vil have

# Udskriv resultaterne
for i, doc in enumerate(results, 1):
    print(f"\n--- Match {i} ---")
    print("📄 Indhold:")
    print(doc.page_content[:500])  # begræns længden
    print("🧾 Metadata:", doc.metadata)