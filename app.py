import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import re

from dotenv import load_dotenv
load_dotenv()


# Load vector store
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)

# med source. Tager måske lang tid?
#qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=db.as_retriever(), return_source_documents = True)


# UI
new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">New image</p>'
LOGO = "C:/Users/Jools/Desktop/RAGmodel hackathon/RIG-logo.png"
st.logo(LOGO, size="large", link=None, icon_image=None)
st.markdown('<h2 style="color:#008567; font-size: 42px; font-family:Arial">Velkommen til Pennyworth</h2>', unsafe_allow_html=True)
st.markdown('<h2 style="color:#008567; font-size: 20px; font-family:Arial">Rigsrevisionens RAG-assistent</h2>', unsafe_allow_html=True)
st.markdown('<p style="font-style:italic; color:#5f5f5f; font-size: 15px; font-family:Arial"> Jeg ved alt om offentliggjorte beretninger og notater.</p>', unsafe_allow_html=True)
topic = st.text_input(":man_in_tuxedo: Hvad kan jeg hjælpe med?")
#timeline_width = st.slider("Er der en bestemt tidsperiode, jeg skal søge i?", min_value = 2010, max_value=2025, )
search_width = st.slider("🔍 Hvor meget materiale skal jeg inddrage i søgningen? (1-10)" , min_value=1, max_value=10, value=4)
attach_sources = st.selectbox(":bookmark_tabs: Tilknyt kilder?", ["Ja", "Nej"])

if st.button("Generér svar"):
    if topic:

        # Internal prompt to guide the model — NOT shown to the user
        prompt = f"""
        Du er en klog embedsmand i staten. Besvar følgende spørgsmål baseret på dokumenterne:

        - Svar på dansk.
        - Skriv tydeligt og præcist.
        - Målgruppen er professionelle fagfolk med en samfundsfaglig uddannelse.
        - Lad være med at opfinde oplysninger: alt skal være korrekt og kunne dokumenteres.
        - Brug gerne punktopstilling

        Det er vigtigt at skelne mellem dokumenttyper. Der er følgende dokumenttyper:
        - Beretninger er hoveddokumenterne, som er Rigsrevisionens undersøgelser af myndigheder i staten
        - "18,4 notater" er opfølgninger, som tager udgangspunkt i ministeriets plan for at rette op på kritikpunkerne i beretningen
        - Fortsatte notater er løbende opfølgninger på beretningerne, som undersøger om ministeriet implementerer kritikpunkterne løbende. Her kan kritikpunkter, opfølgningspunkter, både fortsat være åbne (som betyder at de ikke er afsluttet endnu) eller lukket (som betyder at ministeriet har rettet op på kritikpunkterne)

        Beretninger har typisk denne struktur:
        - Først er der 1-2 forsider og en karakterskala over kritikniveauaet 
        - Så kommer Statsrevisorernes bemærkninger
        - Senere kommer en indledning
        - Så kommer hovedkonklusionen
        - I midten kommer analysekapitlerne
        - til sidst er der et metodebilag
        
        Det er vigtigt at skelne mellem, hvornår et dokument er udgivet, og hvilken periode analysen i dokumentet vedrører.
        Som udgangspunkt er Rigsrevisionen afsenderen på alle dokumenter.

        Spørgsmål: {topic}
        """

        retriever = db.as_retriever(search_kwargs={"k": search_width})

        if attach_sources == "Ja":
            qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature =0.0), retriever=retriever, return_source_documents = True)
        
        else:
            qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature =0.0), retriever =retriever)

        with st.spinner("Genererer svar..."):
            response = qa_chain.invoke(prompt)
        
        answer = response["result"]
        
        st.markdown('<h3 style="color:#5f5f5f; font-size: 20px; font-family:Arial">🧐 Her er hvad jeg kunne finde:</h3>', unsafe_allow_html=True)
        st.text(answer)

        if attach_sources == "Ja":
            sources = response["source_documents"]

            st.markdown('<h3 style="color:#5f5f5f; font-size: 20px; font-family:Arial">📜 Kilder</h3>', unsafe_allow_html=True)
            for i, doc in enumerate(sources, 1):
                source = doc.metadata.get("filename", "Ukendt fil") #så første option hvis den har den, anden hvis ikke
                page = doc.metadata.get("page_number", "?")
                
                snippet = doc.page_content[:350].replace("\n", " ").strip()
                snippet = re.sub("([a-zA-Z])- ", r"\1", snippet)
                st.markdown(f"**{i}. {source}, side {page}**")
                st.markdown(f"_{snippet}..._")

    else:
        st.warning("Indtast venligst et spørgsmål.")
