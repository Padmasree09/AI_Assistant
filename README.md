# Agentic AI Research Assistant

A modular, local-first AI assistant that combines retrieval-augmented generation (RAG), multi-agent orchestration, and LLM-driven reasoning.

This project goes beyond a basic "retrieve then answer" pipeline by introducing query routing, planning, specialized agents, memory, caching, and answer critique inside a single offline-capable architecture.

## Overview

Most RAG systems follow a simple flow:

```text
Query -> Retrieve -> Generate -> Answer
```

This project expands that into a more capable system:

```text
Query -> Route -> Plan -> Execute -> Retrieve -> Generate -> Critique -> Answer
```

The goal is to model a more realistic AI application architecture rather than a single-call chatbot or a minimal RAG demo.

## Architecture

```text
User Query
   |
   v
FastAPI (/query)
   |
   v
Orchestrator
   |
   v
Query Router
   |
   v
Planner
   |
   v
Specialized Agents
   |
   v
Retriever (Qdrant)
   |
   v
LLM Service (llama.cpp)
   |
   v
Critic
   |
   v
Final Answer
```

## Core Components

### Orchestrator

Coordinates the full workflow, including routing, planning, agent execution, retrieval, and validation.

### Query Router

Acts like a Mixture-of-Experts style dispatcher by classifying the incoming query and selecting the most relevant agent path.

Supported intent categories include:

- Research
- Summary
- QA
- Reasoning

### Planner

Breaks complex requests into smaller execution steps.

Example:

```text
Compare RNN vs Transformer
-> Step 1: Retrieve RNN information
-> Step 2: Retrieve Transformer information
-> Step 3: Compare trade-offs
```

### Agents

Each agent is specialized through prompt design and task boundaries.

| Agent | Role |
| --- | --- |
| Research | Deep explanations |
| Summary | Concise summaries |
| QA | Direct factual answers |
| Reasoning | Comparisons and analysis |

### Retriever

Provides semantic search over indexed knowledge using vector embeddings and Qdrant. The design allows agents to perform iterative retrieval when needed.

### LLM Layer

Runs through `llama.cpp` using a local GGUF model such as Phi-3 Mini.

### Critic

Performs a validation pass on generated responses for:

- grounding
- relevance
- completeness

This creates a basic self-reflection loop before returning the final answer.

### Memory

Stores recent context so the system can support follow-up questions more naturally.

### Cache

Avoids recomputation for repeated requests and helps improve response latency.

## Tech Stack

- FastAPI
- Qdrant
- Sentence Transformers
- llama.cpp
- Phi-3 Mini (GGUF)

## Project Structure

```text
ai_assistant/
|-- .env.example
|-- main.py
|-- requirements.txt
|-- agents/
|-- core/
|-- models/
|-- router/
|-- scripts/
`-- services/
```

## Requirements

- Python 3.10+
- Docker, if you want to run Qdrant in a container
- A local GGUF model file for `llama.cpp`
- A built or downloaded `llama.cpp` server binary
- Local access to the embedding model used by `sentence-transformers`

## Configuration

This project is designed so the source code stays in Git while large runtime assets stay outside the repository.

The app reads runtime settings from environment variables. Copy `.env.example` to `.env` and update the values for your machine.

Important variables:

- `APP_HOST`
- `APP_PORT`
- `LLAMA_URL`
- `LLAMA_TIMEOUT_SECONDS`
- `QDRANT_HOST`
- `QDRANT_PORT`
- `QDRANT_COLLECTION`
- `EMBEDDING_MODEL_NAME`
- `LLAMA_SERVER_EXE`
- `LLAMA_MODEL_PATH`

Example external layout:

```text
C:\Users\<you>\ai-projects\
|-- llama.cpp\
`-- models\
```

That way, the repo remains lightweight while still being reproducible for other developers.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Padmasree09/AI_Assistant.git
cd AI_Assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create your local environment file

```bash
copy .env.example .env
```

Edit `.env` so the paths and URLs match your machine.

### 4. Start Qdrant

If Docker is available:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Or on Windows:

```bat
scripts\start_qdrant.bat
```

### 5. Start the local LLM server

Make sure your `.env` points to the correct `llama-server` executable and GGUF model path, then run:

```bat
scripts\start_llm_server.bat
```

### 6. Run the API

```bat
scripts\start_api.bat
```

### 7. Test the API

Health check:

```bash
curl http://localhost:8000/health
```

Query request:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain transformer architecture"}'
```

## Example Flow

For a query like:

```text
Explain transformer architecture
```

The system can:

1. classify the query as research or QA
2. retrieve relevant context from Qdrant
3. generate an answer using the local LLM
4. run a critique step for quality control
5. return the final response

## Key Features

- Multi-agent architecture
- Query routing
- Multi-step planning
- Retrieval-augmented generation
- Local LLM inference
- Critic/self-reflection layer
- Conversational memory
- Response caching
- Offline-capable design

## Design Principles

- Modular architecture
- Separation of concerns
- Replaceable components
- System-level design over framework-heavy coupling

## Trade-offs

| Aspect | Decision |
| --- | --- |
| Model | Smaller local LLM instead of cloud-scale models |
| Latency | Higher due to routing, planning, and critique |
| Accuracy | Improved through retrieval and multi-step reasoning |
| Scalability | Structured so components can be scaled independently |

## Current Limitations

- Quality depends heavily on the local model you run
- Retrieval quality depends on how your vector store is populated
- End-to-end offline behavior assumes all models and services are available locally
- Some advanced behaviors are architecture-oriented and may need further expansion for production use

## Reproducibility Notes

This repository is structured so that a new user can clone the code and reproduce the setup without needing your exact folder layout.

- Source code lives in this repository
- Large runtime assets such as GGUF models and inference binaries stay outside Git
- Runtime behavior is controlled through `.env`
- Helper scripts are included for local startup on Windows

This makes the project easier to share, maintain, and deploy without bloating the repository.

## Why This Project Matters

This project demonstrates:

- AI system design, not just model usage
- modular orchestration across multiple components
- practical trade-offs between latency, accuracy, and cost
- how local AI infrastructure can be composed into a larger assistant workflow

It is a useful portfolio project for showing applied understanding of RAG systems, agent-based design, and local LLM deployment patterns.

## Future Improvements

- Streaming responses
- Hybrid retrieval (BM25 plus semantic search)
- Chat UI
- Multi-model support
- Better evaluation and observability
- Distributed deployment options

## Author

Built as part of hands-on learning in AI systems engineering, with a focus on:

- LLM infrastructure
- RAG systems
- agent-based architectures

## Support

If you find this project useful, consider giving it a star.
