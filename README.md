# AIrail - AI-Powered Indian Railways Assistant 🚆

AIrail is a modern, full-stack conversational AI application designed to provide users with real-time Indian Railways information. It uses a multi-agent LangGraph architecture to seamlessly route user queries to the appropriate tools, fetching live train status, PNR updates, and IRCTC policy information.

---

## 🌟 Key Features

* **Intelligent Routing (Supervisor Agent):** Automatically classifies user queries to route them to the specialized agent (RailRadar, PRS, or RAG).
* **Live Train Tracking (RailRadar Agent):** 
  * Real-time train running status and delays.
  * Train schedules and routes.
  * Live station arrival/departure boards.
  * Smart station lookup resolving city names to official station codes using a local ~7000+ station database.
* **PNR Status (PRS Agent):** Live PNR status tracking and passenger booking details via NTES integration.
* **Knowledge Retrieval (RAG Agent):** Ask questions about IRCTC policies, refund rules, and tatkal bookings. Powered by Qdrant vector search over official IRCTC documents.
* **Modern Interface:** A sleek, dark-themed Glassmorphism UI built with React, featuring Markdown rendering for tables and lists.
* **Secure Authentication:** JWT-based user registration and login with PostgreSQL storage.

---

## 🛠️ Tech Stack

### Frontend
* **Framework:** React + Vite
* **Styling:** Vanilla CSS (Glassmorphism design aesthetic)
* **Markdown:** `react-markdown` with `remark-gfm`
* **Routing:** React Router

### Backend
* **Framework:** FastAPI (Python)
* **AI & Orchestration:** LangGraph, LangChain, Groq API (Llama 3 / Mixtral models)
* **Database (Relational):** PostgreSQL with SQLAlchemy (async)
* **Database (Vector/RAG):** Qdrant (Hybrid search + FlashRank reranking)
* **Embeddings:** FastEmbed (Sentence-Transformers)

---

## 🏗️ Graph Architecture & Agents

AIrail utilizes a multi-agent workflow orchestrated by **LangGraph**. The system is represented as a state graph where the user's query is routed to the most capable expert agent.

```mermaid
graph TD
    User((User Input)) --> ChatAPI[FastAPI /chat]
    ChatAPI --> Supervisor{"Supervisor Agent<br>(Intent Classifier)"}
    
    Supervisor -- "intent == 'greeting'" --> Output((Response))
    Supervisor -- "intent == 'railradar'" --> RailRadar["RailRadar Agent<br>(Live Status & Tracking)"]
    Supervisor -- "intent == 'prs'" --> PRS["PRS Agent<br>(PNR & Booking Status)"]
    Supervisor -- "intent == 'rag'" --> RAG["RAG Agent<br>(IRCTC Policies & FAQ)"]
    Supervisor -- "intent == 'guardrail'" --> Guardrail["Guardrail Agent<br>(Safety & Policy)"]
    
    RailRadar <--> Tools1["RailRadar Tools<br>- Live Status<br>- Search Trains<br>- Schedules<br>- Station DB"]
    PRS <--> Tools2["PRS Tools<br>- NTES Client"]
    RAG <--> Tools3["Qdrant Retriever<br>- Dense + Sparse Search<br>- FlashRank"]
    
    RailRadar --> Output
    PRS --> Output
    RAG --> Output
    Guardrail --> Output
```

### The Agents

#### 1. Supervisor Agent 🚦
The Supervisor acts as the intelligent router for the system. It receives the user's raw message and uses a lightweight LLM call to classify the intent. It strictly routes the query to:
*   `greeting`: Handled inline by the Supervisor for quick, warm responses to conversational openers without routing to a sub-agent.
*   `railradar`: For queries regarding live train running status, train schedules, station boards, and finding trains between stations.
*   `prs`: For checking the status of a specific 10-digit PNR number.
*   `rag`: For general inquiries about baggage rules, ticket cancellation policies, tatkal booking, and other IRCTC FAQs.
*   `guardrail`: For malicious, abusive, or off-topic queries, which are routed to the Guardrail Agent.

#### 2. RailRadar Agent 🚆
The RailRadar agent is a ReAct (Reasoning + Acting) agent responsible for live train tracking. It has access to several specialized tools:
*   `lookup_station_by_name`: A local search tool that intelligently maps human-readable station names (e.g., "Ahmedabad") to official railway station codes (e.g., "ADI") using a ~7000+ entry JSON database. Prioritizes exact matches and major junctions.
*   `get_live_train_status`: Fetches real-time train location, delays, and upcoming stops.
*   `search_trains_between`: Discovers all trains operating between a specific origin and destination.
*   `get_train_schedule`: Retrieves the full stop-by-stop schedule for any train.
*   `get_live_station_board`: Shows live arrivals and departures for a given station.

#### 3. PRS (Passenger Reservation System) Agent 🎫
This agent handles booking-specific queries. 
*   It utilizes the `NTESClient` to fetch real-time PNR statuses.
*   Automatically parses and explains the charting status, passenger waitlist/confirmation status, and coach details.

#### 4. RAG (Retrieval-Augmented Generation) Agent 📚
The RAG agent answers policy-related queries using official IRCTC documentation (PDFs).
*   **Ingestion:** IRCTC policy PDFs are chunked and embedded using FastEmbed (`all-MiniLM-L6-v2` for dense, `Qdrant/bm25` for sparse).
*   **Retrieval:** Uses Qdrant for hybrid search (Reciprocal Rank Fusion) and FlashRank (`ms-marco-TinyBERT-L-2-v2`) for reranking the top results.
*   **Generation:** Feeds the top retrieved, highly relevant document chunks into the LLM (without arbitrary token caps) to produce detailed, accurate, and fully cited answers formatted in Markdown.

#### 5. Guardrail Agent 🛡️
This agent is responsible for handling queries that fall outside the application's scope or violate usage guidelines.
*   Safely blocks prompt injection attempts, abusive language, and off-topic questions.
*   Provides a firm, professional refusal while inviting the user to ask a valid, railways-related question.

---

## 🚀 Getting Started

### Prerequisites
* **Node.js** (v18+)
* **Python** (3.10+)
* **PostgreSQL** database running locally or remotely.
* **Qdrant** running locally (Port 6334 for gRPC).
* API Keys for Groq and RailRadar.

### 1. Backend Setup

Navigate to the backend directory:
```bash
cd backend
```

Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Set up your environment variables. Create a `.env` file in the `backend` directory:
```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/airail
GROQ_API_KEY=your_groq_api_key
RAILRADAR_API_KEY=your_railradar_api_key
JWT_SECRET=your_jwt_secret
```

Start the FastAPI server:
```bash
uvicorn app.main:app --reload --port 8000
```
*Note: On first startup, the backend will automatically chunk and ingest the PDFs located in `backend/data/` into Qdrant.*

### 2. Frontend Setup

Open a new terminal and navigate to the frontend directory:
```bash
cd frontend
```

Install dependencies:
```bash
npm install
```

Start the Vite development server:
```bash
npm run dev
```
The application will be available at `http://localhost:5173`.

---

## 📁 Directory Structure

```
AIrail/
├── backend/
│   ├── app/
│   │   ├── agents/      # LangGraph supervisor, rag, prs, and railradar agents
│   │   ├── models/      # SQLAlchemy DB models and Pydantic schemas
│   │   ├── rag/         # Qdrant ingestion, retrieval, and embeddings logic
│   │   ├── routers/     # FastAPI endpoints (auth, chat, manual API endpoints)
│   │   └── tools/       # Tools for agents (RailRadar, NTES PNR)
│   ├── data/            # IRCTC PDFs for RAG ingestion
│   ├── railwayStationsList.json  # Offline station name-to-code mapping
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/  # React components (MessageBubble, Sidebar, etc.)
│   │   ├── pages/       # Login, Register, and Chat pages
│   │   ├── App.jsx
│   │   └── index.css    # Global Glassmorphism and Markdown styles
│   └── package.json
└── README.md
```

## 📜 License
This project was built for educational and demonstration purposes. Data provided via external APIs is subject to their respective terms of service.
