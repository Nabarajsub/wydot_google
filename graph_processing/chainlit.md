# 🛣️ WYDOT Assistant: GraphRAG Edition

Welcome to the advanced WYDOT Chatbot! This system uses a **knowledge graph** and **multi-agent reasoning** to provide precise answers about transportation specifications and policies.

## 🧠 Key Concepts

### 1. Agentic Reasoning
In "Agentic" mode, the chatbot doesn't just search for text; it acts as an **Orchestrator**. It evaluates your question, decides which specialized "tools" or "agents" to call, and cross-references its findings to ensure accuracy. It can "double-check" its own work before giving you an answer.

### 2. Multi-hop Reasoning
Standard AI often looks at one document at a time. **Multi-hop** allows the AI to "hop" across relationships in our Knowledge Graph. It can connect a person to a project, a project to a specific material specification, and that material to a testing requirement—even if they are in different manuals.

---

## 🛠️ How to Use This Chatbot

1.  **⚙️ Toggle Advanced Modes**: Click the **Settings** icon in the left sidebar to enable **"Multi-Agent Mode"** or **"Multi-hop Reasoning."**
2.  **📄 Analyze Documents**: Upload PDFs, bridge plans, or construction site photos. The AI can "see" and "read" these files to answer specific questions.
3.  **🎙️ Voice Input**: Click the microphone icon to speak your question directly.
4.  **🔗 Cite Your Sources**: The bot will provide `[SOURCE_X]` links. Click them to see exactly which page of the manual the information came from.

---
> [!TIP]
> Use Agentic mode for complex reasoning tasks, and standard mode for quick lookups to save time and tokens.
