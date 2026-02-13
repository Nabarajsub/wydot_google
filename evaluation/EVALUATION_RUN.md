# How to Run the RAG Evaluation Pipeline

## Local (one-off or dev)

1. **Environment**  
   From the **project root** (e.g. `wydot_cloud`), ensure `.env` has:
   - `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`  
   - Optional: `GCP_PROJECT_ID` for full Vertex AI–based metrics

2. **Dataset**  
   Put your eval set in JSONL form (one JSON object per line), e.g.  
   `evaluation/datasets/wydot_golden_sample.jsonl`:
   - `question` (required)
   - `reference_answer` or `answer` (optional)
   - `id` (optional)

3. **Run**  
   From project root:
   ```bash
   cd evaluation
   pip install -r requirements.txt
   python run_local.py
   ```
   Or with options:
   ```bash
   python run_local.py --dataset datasets/wydot_golden_sample.jsonl --output-dir reports
   ```

4. **Output**  
   Report is written to `evaluation/reports/local_report_<timestamp>.json` (or the path you pass with `--output`).

---

## Scheduled / Cloud (Cloud Run Job)

- The pipeline is wired to run as a **Cloud Run Job** (see `evaluation/run_evaluation_job.py` and `.github/workflows/deploy-evaluation.yml`).
- Trigger: **Cloud Scheduler** (e.g. weekly) or manual run:
  ```bash
  gcloud run jobs execute wydot-rag-evaluation --region=REGION
  ```
- Dataset can be provided via env (e.g. `JSONL_PATH` or GCS path); results go to BigQuery and/or GCS (e.g. `wydot-evaluations-*` bucket).

---

## Summary

| Goal              | Command / Trigger                                      |
|-------------------|--------------------------------------------------------|
| Run once locally  | `cd evaluation && python run_local.py`                |
| Custom dataset    | `python run_local.py --dataset path/to/file.jsonl`     |
| Run in cloud      | Execute Cloud Run Job (Scheduler or `gcloud ... execute`) |
