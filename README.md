# LalimSalita — Streamlit deployment

Adapted from `OMG_FinalProject.ipynb`. Uses the same multilingual-E5 embeddings, PDF page extraction, per-page RecursiveCharacterTextSplitter, Chroma cosine retrieval (top 3), and Groq grounded-answer prompt.

## Run locally

1. Install Python 3.11 (recommended).
2. Open a terminal in this folder and run `python -m pip install -r requirements.txt`.
3. Create `.streamlit/secrets.toml` with:
   ```toml
   GROQ_API_KEY = "YOUR_GROQ_API_KEY"
   ```
4. Run `streamlit run app.py`.

The first startup downloads `intfloat/multilingual-e5-small` and generates the vector index in memory. Subsequent reruns in the same process reuse it. It is not a persistent Chroma checkpoint.

## Deploy to Streamlit Community Cloud

1. Upload `app.py`, `rag_backend.py`, `requirements.txt`, and `data/Talasalitaan Aralin 1-10.pdf` to a GitHub repository. Do NOT upload `.streamlit/secrets.toml`.
2. At https://share.streamlit.io select **Create app** and connect the repository, branch, and `app.py` entrypoint.
3. In app settings / Advanced settings / Secrets, add `GROQ_API_KEY = "..."`.
4. Deploy. The initial index creation can take a few minutes and needs sufficient RAM. If the Community Cloud resource limit is exceeded, use a more capable host later.

## Experiment ID caution

The uploaded notebook defines CFG1=300/50, CFG2=500/100, CFG3=800/150. Prior evaluation conversations used a different naming scheme. This deployment uses **500/100**, not a claim about which notebook CFG ID was selected. To change the live configuration, set environment variables `CHUNK_SIZE` and `CHUNK_OVERLAP` and restart the app. This does not reproduce earlier benchmark scores automatically.

## Security

Keep the Groq API key private. Do not commit it to GitHub. Public deployments may consume your API quota when visitors submit questions.
