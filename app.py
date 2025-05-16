import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import pandas as pd
import os

from utils.vector_store import load_docs_to_vectorstore
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.title("🔋 Smart Factory Energy Optimizer")

csv_path = r"data/manufacturing_data.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.subheader("📊 Uploaded Energy Data")
    st.write(df.head())
else:
    st.error(f"CSV file not found at: {csv_path}")

pdf_path = r"documents/iso50001_guidelines.pdf"
if os.path.exists(pdf_path):
    try:
        vectorstore = load_docs_to_vectorstore(pdf_path)

        model_name = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

        llm = HuggingFacePipeline(pipeline=pipe)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

        st.subheader("🔍 Ask a Question")
        query = st.text_input("Ask something related to energy optimization or the document:")
        if query:
            result = qa_chain.run(query)
            st.success("Answer:")
            st.write(result)

    except Exception as e:
        st.error(f"Error loading PDF into vectorstore: {e}")
else:
    st.warning(f"PDF file not found at: {pdf_path}")

st.markdown("---")
st.subheader("📈 Business Value")
st.write("""
By using AI-powered analysis of energy data and energy-saving guidelines (ISO 50001),
this tool helps factories reduce power waste, detect anomalies, and optimize shift schedules,
potentially cutting energy costs by up to 12% per unit.
""")
