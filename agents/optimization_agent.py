from utils.rag_utils import ask_rag

def optimize_energy(query, vectorstore):
    return ask_rag(query, vectorstore)
