# House Price Prediction

An end-to-end machine-learning application that predicts California median house
values from block-group features. A tuned **HistGradientBoostingRegressor** pipeline
is served by a **FastAPI** backend and consumed by a **React (Vite)** multi-page frontend.

---

## 🧠 Authorship & Contribution

- **Machine learning (mine):** the modeling work is my own — exploratory data analysis,
  preprocessing pipeline design (`ColumnTransformer` with imputation, scaling, and
  one-hot encoding), model benchmarking, `GridSearchCV` hyperparameter tuning, evaluation,
  and the production training script. This is the part I focus on and can explain in depth.
- **Frontend & API wiring (AI-assisted):** the React UI and the FastAPI serving layer were
  scaffolded with AI assistance, then reviewed and adapted by me. They are a thin presentation
  and serving layer over the model — the ML is the substance of the project.

---

## 📊 The Model

- **Dataset:** California Housing (20,640 block groups, 9 features).
- **Pipeline:** median imputation + standard scaling for numeric features; most-frequent
  imputation + one-hot encoding for `ocean_proximity`; all baked into a single scikit-learn
  `Pipeline` so the API can send raw values.
- **Algorithm:** `HistGradientBoostingRegressor`, selected over Linear/Ridge/Lasso/RandomForest
  by 5-fold cross-validated RMSE, then tuned with `GridSearchCV`.
- **Performance:** ~0.82 test R² (RMSE ≈ $49k). The target is capped at $500k in this dataset,
  which limits achievable error at the top end.
- **Top features (permutation importance):** `median_income`, then geographic position
  (`longitude`/`latitude`) and `ocean_proximity` — consistent with how California housing prices
  actually behave.

---

## 🏗️ Architecture

```
backend/   FastAPI + scikit-learn
  training.py   load → split → preprocess → train (tuned HGB) → evaluate → save pipeline
  main.py       /health, /predict, /dataset-stats, /model-info
frontend/  React + Vite (multi-page)
  src/api.js              single API base URL + fetch helpers
  src/pages/Predict.jsx        9-input form → /predict
  src/pages/DatasetStats.jsx   EDA dashboard → /dataset-stats
  src/pages/ModelInsights.jsx  permutation importances → /model-info
```

The frontend never loads the model — everything goes through the API contract.
Swapping the model requires no frontend changes.

---

## ▶️ Running Locally

**Backend** (from `backend/`):
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# copy .env.example to .env and set PROJECT_ROOT to the absolute backend path
# place housing.csv in backend/dataset/
python training.py        # trains + saves the model
uvicorn main:app --reload # serves on http://127.0.0.1:8000
```

**Frontend** (from `frontend/`, in a second terminal):
```bash
npm install
npm run dev               # serves on http://localhost:5173
```

---

## 🛠️ Tech Stack

**ML:** Python, scikit-learn, pandas, NumPy
**Backend:** FastAPI, Uvicorn
**Frontend:** React, Vite, React Router