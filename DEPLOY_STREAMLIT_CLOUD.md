# 🌐 Deploy to Streamlit Community Cloud (free public link)

This gives you a URL you can send to anyone. They click it and the dashboard
opens in their browser — **no install, no Python, nothing to download.** This is
the easiest possible experience for a non-technical recipient.

## One-time setup (about 5 minutes)

1. **Push this project to GitHub** (public or private repo). From this folder:

   ```bash
   git add .
   git commit -m "Add easy-run launchers and deployment config"
   git push
   ```

   > Make sure `venv/` is NOT pushed — it's already in `.gitignore`, so you're fine.

2. Go to **https://share.streamlit.io** and sign in with GitHub.

3. Click **"Create app"** → **"Deploy a public app from GitHub"** and choose:
   - **Repository:** your repo
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`

4. Click **"Advanced settings"** and set **Python version: 3.11**.
   (This matters — the saved model needs 3.11 + the pinned `requirements.txt`.)

5. Click **Deploy**. After a few minutes you'll get a link like
   `https://your-app-name.streamlit.app`.

6. **Send that link** to the person who needs the app. Done.

## Notes

- The included `requirements.txt` is already pinned to versions known to work, so
  the deploy should "just work."
- The trained model (`models/churn_model.pkl`) and SHAP images (`reports/`) are
  committed to the repo, so the dashboard works immediately online.
- Whenever you push changes to GitHub, the live app updates automatically.
