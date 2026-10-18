# Invyra Business Intelligence AI

**AI-powered business insights and report analysis for Invyra POS.**

Invyra Business Intelligence AI is an AI-powered assistant designed to help business owners understand their business reports using natural-language questions. Instead of manually reviewing multiple reports, users can upload supported documents, ask questions, explore business trends, and make more informed decisions using the information available in their data.

The project uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded documents and generate context-aware answers.

## Overview

Business owners often have to review sales reports, inventory summaries, purchase records, expense statements, and other business documents to understand how their business is performing.

Invyra Business Intelligence AI aims to simplify this process by allowing users to interact with their reports conversationally.

For example, users could ask:

- What were my best-performing products this month?
- Which products have declining sales?
- What are the major expenses in this report?
- Which products should I review for low stock?
- How has my business performance changed compared with the previous period?
- What actions could help improve my sales?
- Summarize the key findings from these reports.

Answers and recommendations should be grounded in the uploaded data. Questions that require missing information or calculations beyond the available data should be identified clearly rather than answered with unsupported assumptions.

## Key Features

- **Document-based AI Q&A:** Ask natural-language questions about uploaded reports.
- **Retrieval-Augmented Generation (RAG):** Retrieve relevant document content before generating answers.
- **Semantic search:** Find information based on meaning, not only exact keyword matches.
- **Hybrid search:** Combine semantic retrieval and keyword-based search for relevant results.
- **Source references:** Trace answers back to relevant document excerpts where supported.
- **PDF processing:** Extract text from supported PDF documents.
- **OCR support:** Process scanned PDFs when the required OCR dependencies are configured.
- **Business insights:** Explore report findings and identify potential opportunities for improvement.
- **FastAPI backend:** Expose document-ingestion and question-answering endpoints.
- **PostgreSQL with pgvector:** Store document information and perform vector similarity searches.
- **Testing:** Support automated testing of retrieval and document-processing logic.

*Note: The features above describe the underlying project capabilities and intended Invyra use case. Actual availability depends on the code, configuration, and integrations in your current version.*

## Technology Stack

| Component | Technology |
|---|---|
| Backend API | Python, FastAPI |
| Database | PostgreSQL |
| Vector search | pgvector |
| Retrieval | Semantic and keyword search |
| AI and embeddings | OpenAI API |
| Document processing | pypdf, pdf2image, Tesseract OCR |
| Database access | SQLAlchemy, psycopg |
| Testing | pytest |
| Development environment | Docker, Docker Compose |

## How It Works

1. **Upload:** The user uploads a supported business report.
2. **Extract:** The application extracts readable content from the document, using OCR when configured and required.
3. **Process:** Extracted content is split into smaller text sections.
4. **Index:** The sections are converted into embeddings and stored for retrieval.
5. **Ask:** The user submits a natural-language question.
6. **Retrieve:** The system searches for relevant information using configured retrieval methods.
7. **Generate:** The AI generates an answer based on the retrieved context.
8. **Review:** The user can review the answer and its available source references.

## Planned Invyra Integration

The long-term goal is to connect this assistant with Invyra's POS and inventory reporting workflows.

Potential capabilities include:

- Uploading and analyzing sales, purchase, inventory, expense, and profit-and-loss reports.
- Comparing business performance across selected periods.
- Identifying sales trends and products that may need attention.
- Summarizing important business metrics in plain language.
- Generating practical suggestions based on available business data.
- Supporting business-specific questions while keeping each business's data isolated.

These integrations require additional implementation and validation before they can be considered production features.

## Getting Started

The following instructions are based on the upstream project's setup. Verify the repository's current configuration before running them.

### Prerequisites

- Python 3.11 or later
- Docker and Docker Compose
- An OpenAI API key
- Required PDF/OCR system dependencies if OCR functionality is needed

### Installation

Clone your repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd invyra-business-intelligence-ai
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install the project dependencies using the repository's supported installation method. For the upstream project, this is:

```bash
pip install -e ".[dev]"
```

Configure environment variables:

```bash
cp .env.example .env
```

On Windows Command Prompt, copy the example file using:

```bat
copy .env.example .env
```

Update `.env` with your development configuration, including the required API key and database connection.

Start the database services:

```bash
docker compose up -d
```

Start the FastAPI application using the module path configured in your repository. The upstream project uses:

```bash
uvicorn ai_search.main:app --reload
```

Open the local application at:

```text
http://127.0.0.1:8000/
```

**Important:** If you rename Python packages or change the project structure, update the installation and startup commands accordingly.

## Testing

Run the project's automated tests using:

```bash
pytest -q
```

The tests and their coverage depend on the test suite included in the current repository.

## Security Considerations

Business reports may contain confidential sales, customer, supplier, and financial information.

Before using this project with real Invyra customer data:

- Implement authentication and authorization.
- Enforce strict business-level data isolation on every document, search, and conversation request.
- Validate file types, file sizes, and uploaded filenames.
- Keep API keys and database credentials in environment variables.
- Configure HTTPS, rate limiting, logging, and appropriate error handling.
- Avoid logging sensitive business information unnecessarily.
- Define document retention and deletion policies.
- Review how uploaded data is sent to external AI providers.
- Test authorization boundaries before deployment.

This project should not be treated as production-ready for customer financial data until the relevant security controls have been implemented and verified.

## Project Status

**Development in progress.**

The goal is to build an AI-powered report analysis assistant for Invyra, evolving from document question-answering toward useful, data-grounded business intelligence.

## Acknowledgements

This project is based on and adapted from the open-source project [ai-search-rag](https://github.com/temurjonm/ai-search-rag) by [temurjonm](https://github.com/temurjonm).

Please review the upstream repository's license and comply with its attribution, redistribution, and modification requirements before publishing or distributing this project.

## Disclaimer

AI-generated insights are intended to support business analysis, not replace verified financial reports or professional accounting advice. Always validate important financial figures and business decisions against the original records.
