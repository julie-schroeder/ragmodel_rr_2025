### Formål
# Scriptet laver pdf'er om til tekst chunks med metadata og gemmer dem så i et jsonl-format
#Man skal indsætte en sti i linje 11

# setup
import os # for at tilgå mapper
import fitz # for at læse pdfer
from langchain.text_splitter import RecursiveCharacterTextSplitter # for at chunke
import json # for at eksportere til json

pdf_folder = "" #indsæt lokal sti her

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100) # langchain configuration

#funktion, der tildeler dokumenttype
def klassificer_dokumenttype(filename):
    fname = filename.lower()
    
    if "18,4 notat" in fname:
        return "18,4 notat"
    if "fortsat notat" in fname:
        return "fortsat notat"
    if "forsat notat" in fname:
        return "forsat notat"
    else:
        return "beretning"
    

docs = os.listdir(pdf_folder)
all_chunks = []

for pdf in docs:

    filepath = os.path.join(pdf_folder, pdf)   # definerer objektet jeg vil hente

    doc = fitz.open(filepath) # bruger fitz til at hente og ekstrahere pdf'en

    for i, page in enumerate(doc, start = 1):
        dokumenttype = klassificer_dokumenttype(pdf)
        
        page_text = page.get_text()
        
        aar = pdf[-8:-4]
        
        beretning = pdf[:6]

        metadata = {
            "filename": pdf,
            "page_number": i,
            "dokumenttype": dokumenttype,
            "aar": int(aar),
            "beretning": beretning
        }
        
        chunks = splitter.create_documents([page_text], [metadata])
        all_chunks.extend(chunks)


with open("json_database/rag_chunks.jsonl", "w", encoding="utf-8") as f:
    for doc in all_chunks:
        json.dump({
            "content": doc.page_content,
            "metadata": doc.metadata
            }, f)
        f.write("\n")
