
# 🧠 MargDarshan

### AI-Powered Multi-Agent Personalized Learning Path Generator

MargDarshan is a **multi-agent learning planner** built using **LangGraph**, **Groq LLM**, and **Tavily Search**.

It generates a **personalized 4-week learning roadmap** from:

* Any learning topic
* Or an uploaded PDF syllabus

The system first diagnoses skill gaps using a **human-in-the-loop MCQ assessment**, then dynamically orchestrates multiple AI agents to build and refine the roadmap.

---

## 🚀 Core Architecture

This project implements a **true agentic AI workflow**:

* 🧠 LLM-powered Orchestrator (Groq – LLaMA 3.3 70B)
* 🔄 Cyclical multi-agent execution (LangGraph StateGraph)
* 🌐 Web resource discovery (Tavily Search)
* 📊 Skill assessment & gap detection
* 🔍 Self-reflection & quality control loop
* 💬 AI Mentor chat support
* 🖥 Streamlit interactive UI

---

## 🏗 System Workflow

### 1️⃣ Goal Input

User provides:

* A topic (e.g., Machine Learning)
* OR uploads a syllabus PDF

PDF text is extracted using `pypdf`.

---

### 2️⃣ MCQ Assessment (Human-in-the-loop)

Agent: `MCQ_AGENT`

* Generates 5 MCQs using Groq LLM
* Covers fundamentals, applications, and challenges
* Waits for user answers before proceeding

---

### 3️⃣ Evaluation

Agent: `EVALUATE_AGENT`

* Scores user responses
* Calculates performance (0–5)
* Determines learning level:

  * Beginner
  * Intermediate
  * Advanced

---

### 4️⃣ Skill Gap Detection

Agent: `SKILL_GAP_AGENT`

* Identifies weak concepts
* Lists incorrect question themes
* Creates structured improvement areas

---

### 5️⃣ Resource Research

Agent: `RESEARCH_AGENT`

* Uses Tavily advanced search
* Finds curated tutorials and documentation
* Adapts search depth based on learner level

---

### 6️⃣ Roadmap Generation

Agent: `ROADMAP_AGENT`

Builds structured:

```
Week 1 – Foundation
Week 2 – Core Concepts
Week 3 – Advanced Topics
Week 4 – Practice & Projects
```

Includes:

* Daily breakdown
* Practice tasks
* Resource mapping
* Skill-based adjustments

---

### 7️⃣ Self-Improving Reflection Loop

Agent: `REFLECT_AGENT`

Quality checks roadmap for:

* Clear structure
* Specific topics
* Practical exercises
* Realistic pacing

If not good enough:

* Roadmap regenerates
* Max 2 retries
* Stops when quality passes

---

### 8️⃣ Explainable AI

Agent: `EXPLAIN_AGENT`

Provides transparency:

* Assessment score
* Identified gaps
* Iterations performed
* How roadmap decisions were made

---

### 9️⃣ AI Mentor Chat

Post-generation support:

* Context-aware responses
* Roadmap-based answers
* Encouraging and practical guidance

---

## 🔄 Orchestration Logic

A Groq-powered **LLM Orchestrator** dynamically routes between agents.

Instead of fixed flow:

* The LLM observes system state
* Chooses which agent to run next
* Can re-run agents
* Stops only when roadmap is complete and high quality

This creates a **cyclical agent graph**, not a linear pipeline.

---

## 🛠 Tech Stack

| Component        | Technology                     |
| ---------------- | ------------------------------ |
| LLM              | Groq (LLaMA 3.3 70B Versatile) |
| Agent Framework  | LangGraph                      |
| Web Search       | Tavily API                     |
| UI               | Streamlit                      |
| PDF Parsing      | pypdf                          |
| Tool Integration | LangChain Tools                |
| Type Safety      | TypedDict State                |

---

## 📦 Project Structure

```
final.py        → Complete application (UI + agents + graph)
README.md       → Documentation
```

The entire system is implemented in a single integrated application file.

---

## ⚙ Installation

```bash
git clone <repo_url>
cd <repo_name>
pip install -r requirements.txt
```

Or manually install:

```bash
pip install streamlit langgraph langchain groq tavily-python pypdf
```

---

## 🔐 Environment Setup

Replace API keys in the file:

```python
GROQ_API_KEY = "your_groq_key"
TAVILY_API_KEY = "your_tavily_key"
```

For production, move them to environment variables.

---

## ▶ Run Application

```bash
streamlit run final.py
```

---

## 🎯 Key Features

* Human-in-the-loop adaptive learning
* Multi-agent cyclical architecture
* LLM-based decision routing
* Self-improving quality control
* Explainable AI transparency
* Personalized roadmap generation
* Context-aware mentor chatbot
* PDF syllabus understanding

---

## 🧠 Why This Is Advanced

Unlike simple roadmap generators, this system:

* Diagnoses before prescribing
* Uses dynamic orchestration instead of static pipelines
* Performs automated quality validation
* Integrates external research
* Provides explainability layer
* Maintains conversational mentoring context

This makes it a **true agentic learning planner**, not just a prompt wrapper.

---



