Local RAG Engine (Java 21 & LangChain4j)

A persistent, incremental Retrieval-Augmented Generation (RAG) engine built with Java 21, LangChain4j, and Google Gemini API. It enables semantic search and context-aware question answering over local personal documents (PDF, TXT, Markdown) without requiring external vector database servers.

Architecture & Key Features

Incremental Document Ingestion: Tracks file names and modification timestamps (indexed_files.txt) to vectorize and append only newly added or updated documents.

Local ONNX Embeddings: Uses BgeSmallEnV15EmbeddingModel locally on CPU to generate vector embeddings for English documents with zero embedding API cost.

Persistent Vector Storage: Serializes embeddings to disk (vector_store.json) for instant startup across sessions.

Recursive Chunking: Splits documents into 500-character segments with a 100-character overlap to preserve semantic context across boundaries.

Context-Aware Chat Memory: Maintains a sliding window of the last 10 messages (MessageWindowChatMemory) for natural follow-up questions.

Hallucination Guardrails: Applies a minimum cosine similarity threshold (minScore: 0.6) during retrieval to ensure responses stay grounded in source documents.

Tech Stack

Language: Java 21

Build Tool: Maven

AI Orchestration: LangChain4j (0.36.2)

LLM: Google Gemini (gemini-3.5-flash-lite)

Embedding Model: BAAI bge-small-en-v1.5 (Local ONNX)

Document Parsing: Apache Tika (via langchain4j-easy-rag)

Project Structure

local_rag_engine/
├── documents/              # Local directory for your PDF/TXT/MD files (git-ignored)
├── src/main/java/org/example/
│   └── Main.java           # Ingestion pipeline, vector store management & CLI chat loop
├── .gitignore              # Excludes sensitive keys, local docs, and vector store files
├── pom.xml                 # Maven dependencies
└── README.md


Getting Started

1. Prerequisites

JDK 21 or higher

Maven 3.8+

A free Google Gemini API Key from Google AI Studio

2. Clone the Repository

git clone https://github.com/ffrrkknn/local-rag-engine.git
cd local-rag-engine


3. Configure Environment Variable

Set your Gemini API key as an environment variable (never hardcode it in source files):

Windows (PowerShell):

$env:GEMINI_API_KEY="your_api_key_here"


Linux / macOS:

export GEMINI_API_KEY="your_api_key_here"


(In IntelliJ IDEA: Run -> Edit Configurations -> Environment variables -> Add GEMINI_API_KEY)

4. Add Documents & Run

Place your .pdf, .txt, or .md files inside the documents/ folder in the project root.

Run Main.java via IntelliJ IDEA or Maven:

mvn compile exec:java -Dexec.mainClass="org.example.Main"


On the first run, the engine will chunk and index your documents into vector_store.json. Subsequent runs load the index immediately.

Roadmap

[x] Basic In-Memory RAG Pipeline

[x] Persistent Disk Vector Store & Recursive Chunking

[x] Incremental Document Ingestion

[ ] Modular Package Architecture (ingestion, retrieval, agent)

[ ] Autonomous AI Agent Integration with @Tool Calling