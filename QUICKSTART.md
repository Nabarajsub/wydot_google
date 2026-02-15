# 🚀 WYDOT Assistant: Quickstart Guide

Follow these steps to test the entire application locally from a completely fresh state.

## 1. Clean & Fresh Start
Before starting, ensure any old local data or running containers are removed:
```bash
docker-compose down -v
```
> [!NOTE]
> The `-v` flag deletes local Docker volumes, ensuring the PostgreSQL database starts empty and is seeded fresh.

## 2. Build & Launch
Build the images and start the services in the background:
```bash
docker-compose up --build -d
```
*Wait about 30-60 seconds for the database and embedding models to initialize.*

## 3. Verify Services
The application is ready when the following links are accessible:

| Service | URL | Purpose |
| :--- | :--- | :--- |
| **Chatbot** | [http://localhost:8080](http://localhost:8080) | Main Assistant UI. Use **"Continue as Guest"**. |
| **Monitoring** | [http://localhost:8082](http://localhost:8082) | Admin Dashboard. View real-time metrics & evaluations. |
| **Embeddings** | [http://localhost:8081](http://localhost:8081) | Internal vector serving endpoint. |

## 4. Test "From Beginning" Flow
1. Open [http://localhost:8080](http://localhost:8080).
2. Click the **"Continue as Guest"** button.
3. Ask a question (e.g., *"What are the traffic safety guidelines?"*).
4. Open [http://localhost:8082](http://localhost:8082) in a new tab to see your request appear in the **Monitoring** metrics.

---
**Next Steps**: When you are happy with the local results, refer to [DEPLOYMENT_GUIDE.md](file:///Users/uw-user/.gemini/antigravity/brain/d1b88d7f-ac8b-4daa-97f1-e03dff087bfc/DEPLOYMENT_GUIDE.md) to push to Google Cloud.
