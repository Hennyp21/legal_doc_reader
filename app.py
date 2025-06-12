import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


from langchain.chat_models import ChatOpenAI
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Legal Doc Reader", layout="centered")
st.title(" Legal Document Reader ")

uploaded_file = st.file_uploader("📤 Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("🔄 Reading your document..."):
        reader = PdfReader(uploaded_file)
        text = "".join(page.extract_text() for page in reader.pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # Load both LLMs
    ollama_llm = Ollama(model="mistral")
    
    gpt_llm = ollama_llm
    llm_used = "Mistral (Ollama)"

    qa_chain = RetrievalQA.from_chain_type(llm=gpt_llm, chain_type="stuff", retriever=retriever)

    st.success(f"✅ Document processed! Using: **{llm_used}**")

    # 🔥 Prompt-to-Response (Ollama)
    st.subheader(" Response from Fixed Prompt (Ollama)")

    if st.button("Run Fixed Prompt"):
        with st.spinner("Generating response..."):
            fixed_prompt = (
                "You're a legal assistant. Find all important key sentences in the document "
                "and write them in 50-100 words in separate sentences (new lines). "
                "Also, extract payment info, deadlines, or key data in JSON format. "
                "Keep legal terms but make the sentences easy to understand."
            )
            response = ollama_llm.predict(f"{fixed_prompt}\n\nDocument:\n{text[:5000]}")
        st.success("✅ Ollama's Response:")
        st.write(response)

    # ❓ Q&A Section
    st.subheader(f"❓ Ask a Question ({llm_used})")
    user_question = st.text_input("What do you want to ask about the document?")

    if st.button("🔍 Get Answer"):
        if user_question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("🔎 Thinking..."):
                try:
                    answer = qa_chain.run(user_question)
                    st.success(f"✅ Answer from {llm_used}:")
                    st.write(answer)
                except Exception as e:
                    st.error("❌ LLM failed. Trying fallback model...")
                    answer = Ollama(model="mistral").predict(user_question)
                    st.success("✅ Fallback Answer from Mistral:")
                    st.write(answer)
