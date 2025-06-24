### Formål
# Scriptet indlæser json filerne, embedder dem og gemmer dem så i et vector_store format
# Kræver en API key

#Indlæser Data fra json
import json
import os

documents = []
with open("json_database/rag_chunks.jsonl", "r") as f:
    for line in f:
        item = json.loads(line) #loads fordi det er et jsonl format? Kan ikke huske rationalet her.
        documents.append(item) #henter fra json


from langchain.docstore.document import Document
docs = [Document(page_content=d["content"], metadata=d["metadata"]) for d in documents] #binder det sammen i et langchain dokument


#API del
from dotenv import load_dotenv
load_dotenv() ##henter key

from langchain_openai import OpenAIEmbeddings
import openai
embedding_model = OpenAIEmbeddings()


#Gemmer det embeddede materiale som vector store
from langchain.vectorstores import FAISS 
db = FAISS.from_documents(docs, embedding_model)
print("API key:", os.getenv("OPENAI_API_KEY"))
db.save_local("vector_store")
