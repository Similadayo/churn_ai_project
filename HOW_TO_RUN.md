# ▶️ How to Run This App (Plain-English Guide)

You do **not** need to know any programming. Pick the option that matches you.

---

## ✅ Option 1 — Open a link (easiest, nothing to install)

If the person who sent you this also sent a **web link** (something like
`https://...streamlit.app`), just click it. The dashboard opens in your browser.
There is nothing to install. **You can stop reading here.**

---

## 💻 Option 2 — Run it on your own Windows computer

1. **Unzip** the folder you received (right-click the `.zip` → *Extract All*).
2. Open the unzipped `churn_ai_project` folder.
3. **Double-click the file named `run.bat`.**
4. A black window opens and does some setup automatically.
   - The **first time** this takes a few minutes (it downloads what it needs).
   - After that it's fast.
5. Your web browser opens with the dashboard. 🎉
6. To stop the app, just **close the black window**.

> **If it says "Python was not found":**
> Install Python 3.11 from
> https://www.python.org/downloads/release/python-3119/ —
> during install, **tick the box that says "Add python.exe to PATH"**,
> then double-click `run.bat` again.

---

## 🍎 Option 3 — Run it on a Mac

1. Unzip the folder.
2. Open the **Terminal** app.
3. Type `bash ` (with a space), then **drag the `run.sh` file** into the Terminal
   window and press **Enter**.
4. The dashboard opens in your browser. To stop it, press **Ctrl + C** in Terminal.

---

## 📊 Once the dashboard is open

- **Predictions tab:** click *Browse files* and upload a customer CSV
  (the included `data/customers.csv` works for a demo).
- **Explainability tab:** view the charts explaining *why* customers churn.
- **Trends & Segmentation tab:** see churn broken down by group
  (run a prediction first).

---

## ❓ Still stuck?

Send a photo of the black window / error message to the person who shared this
project. The most common cause is simply not having Python installed yet
(see Option 2 above).
