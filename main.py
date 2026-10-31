"""Invyra in the terminal. Run with: python main.py"""
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

from rag_core import answer_from_documents, answer_with_local_model, set_llm_model
from vector_db import get_retrievers

LLM_MODEL = "qwen3:8b"
EMBED_MODEL = "mxbai-embed-large"

llm = OllamaLLM(model=LLM_MODEL, num_ctx=8192, temperature=0.2)
set_llm_model(llm)

print("Invyra is reading your business files...")
csv_retriever, pdf_retriever, stats = get_retrievers(OllamaEmbeddings(model=EMBED_MODEL))
print(f"Ready: {stats['csv']:,} data records, {stats['pdf']:,} document passages.\n")

print("Choose a source:")
print("1. Business Data (CSV)")
print("2. Business Documents (PDF)")
retriever = pdf_retriever if input("Enter 1 or 2: ").strip() == "2" else csv_retriever

while True:
    print("\n----------------------------- New question -----------------------------")
    question = input("Ask Invyra (q to quit): ").strip()
    if question.lower() == "q":
        break
    if not question:
        continue

    result = answer_from_documents(question, retriever)
    if result["found"]:
        print("\n" + result["answer"])
        if result["sources"]:
            print("\nFrom: " + "; ".join(result["sources"]))
    else:
        print(f"\n(Not found in your business files. Answering with local model {LLM_MODEL}.)\n")
        print(answer_with_local_model(question))