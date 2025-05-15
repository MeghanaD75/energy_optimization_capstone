from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

def ask_rag(query, vectorstore):
    retriever = vectorstore.as_retriever()
    llm = HuggingFaceHub(repo_id="google/flan-t5-large")  # Uses HuggingFaceHub
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(query)
