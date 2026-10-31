"""Invyra data layer: reads business files, stores them in Chroma, returns retrievers."""
import json
import os

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    PyPDFLoader = None
    print("Warning: PyPDFLoader not found. PDF support is off. Run: pip install pypdf langchain-community")

# Input folders
CSV_FOLDER = "local_csv"
PDF_FOLDER = "local_pdf"

# Storage. These are new locations on purpose: your old databases were built with
# whole-page PDFs and a pizza-review CSV layout, so Invyra re-indexes everything once.
# You can delete the old chroma_csv_db / chroma_pdf_db folders and *_files_record.json files.
DB_ROOT = "./invyra_db"
CSV_DB_LOCATION = f"{DB_ROOT}/csv"
PDF_DB_LOCATION = f"{DB_ROOT}/pdf"
CSV_RECORD_PATH = f"{DB_ROOT}/csv_files.json"
PDF_RECORD_PATH = f"{DB_ROOT}/pdf_files.json"

CSV_COLLECTION = "invyra_csv"
PDF_COLLECTION = "invyra_pdf"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_files_recursive(root_folder, extension):
    """Recursively find files with the given extension in root_folder."""
    if not os.path.exists(root_folder):
        print(f"Warning: folder '{root_folder}' not found.")
        return []
    found = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(extension):
                found.append(os.path.join(dirpath, filename))
    return sorted(found)


def load_processed_files(path):
    try:
        if not os.path.exists(path):
            return set()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data) if isinstance(data, list) else set()
    except (json.JSONDecodeError, TypeError, OSError) as e:
        print(f"Could not read {path}: {e}")
        return set()


def save_processed_files(path, processed):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(processed), f)


def store_count(store):
    """Number of items stored in a Chroma collection."""
    try:
        return store._collection.count()
    except Exception:
        return 0


def add_in_batches(store, documents, ids):
    for i in range(0, len(documents), BATCH_SIZE):
        store.add_documents(documents=documents[i:i + BATCH_SIZE], ids=ids[i:i + BATCH_SIZE])


# ---------------------------------------------------------------------------
# CSV: works with ANY columns (sales, customers, feedback, finance...)
# ---------------------------------------------------------------------------
def read_csv_any_encoding(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def csv_to_documents(csv_path, filename):
    df = read_csv_any_encoding(csv_path)
    documents, ids = [], []
    for i, record in enumerate(df.to_dict("records")):
        parts = [f"{col}: {val}" for col, val in record.items()
                 if pd.notna(val) and str(val).strip() != ""]
        if not parts:
            continue
        doc_id = f"{filename}_row{i + 2}"  # +2 = spreadsheet row number (header is row 1)
        documents.append(Document(
            page_content=" | ".join(parts),
            metadata={"source": filename, "row": i + 2},
        ))
        ids.append(doc_id)
    return documents, ids


# ---------------------------------------------------------------------------
# PDF: split pages into small passages so answers stay precise
# ---------------------------------------------------------------------------
def pdf_to_documents(pdf_path, filename):
    loader = PyPDFLoader(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    documents, ids = [], []
    for page_doc in loader.load():
        text = page_doc.page_content.strip()
        if not text:
            continue
        page_number = int(page_doc.metadata.get("page", 0)) + 1
        for j, chunk in enumerate(splitter.split_text(text)):
            documents.append(Document(
                page_content=chunk,
                metadata={"source": filename, "page": page_number},
            ))
            ids.append(f"{filename}_p{page_number}_c{j}")
    return documents, ids


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def index_folder(store, folder, extension, record_path, builder):
    """Adds new files from `folder` to `store`. Returns how many items were added."""
    processed = load_processed_files(record_path)

    # If the database was deleted but the record file was not, start over.
    if processed and store_count(store) == 0:
        processed = set()

    added = 0
    for path in find_files_recursive(folder, extension):
        filename = os.path.relpath(path, folder)
        if filename in processed:
            continue
        try:
            documents, ids = builder(path, filename)
            if not documents:
                print(f"No readable content in {filename} (scanned PDF or empty file?). Skipped.")
                continue
            add_in_batches(store, documents, ids)
            processed.add(filename)
            save_processed_files(record_path, processed)
            added += len(documents)
            print(f"Indexed {filename}: {len(documents)} items")
        except Exception as e:
            print(f"Error indexing {path}: {e}")
    return added


def get_retrievers(embeddings_model):
    """Returns (csv_retriever, pdf_retriever, stats). stats = items stored per source."""
    os.makedirs(DB_ROOT, exist_ok=True)
    os.makedirs(CSV_FOLDER, exist_ok=True)
    os.makedirs(PDF_FOLDER, exist_ok=True)

    csv_store = Chroma(
        collection_name=CSV_COLLECTION,
        persist_directory=CSV_DB_LOCATION,
        embedding_function=embeddings_model,
    )
    index_folder(csv_store, CSV_FOLDER, ".csv", CSV_RECORD_PATH, csv_to_documents)

    pdf_store = Chroma(
        collection_name=PDF_COLLECTION,
        persist_directory=PDF_DB_LOCATION,
        embedding_function=embeddings_model,
    )
    if PyPDFLoader is not None:
        index_folder(pdf_store, PDF_FOLDER, ".pdf", PDF_RECORD_PATH, pdf_to_documents)

    stats = {"csv": store_count(csv_store), "pdf": store_count(pdf_store)}
    csv_retriever = csv_store.as_retriever(search_kwargs={"k": 8})
    pdf_retriever = pdf_store.as_retriever(search_kwargs={"k": 6})
    return csv_retriever, pdf_retriever, stats