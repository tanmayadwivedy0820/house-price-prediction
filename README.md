# House Price Prediction

End-to-end ML app predicting California median house values.
- **backend/**  — Python: scikit-learn pipeline (HistGradientBoostingRegressor) + FastAPI
- **frontend/** — React + Vite (scaffolded separately; see note in folder)

## Setup (backend)
1. `cd backend`
2. Copy `.env.example` to `.env` and set `PROJECT_ROOT` to the absolute backend path.
3. Put `housing.csv` in `backend/dataset/`.
4. `python -m venv .venv` then activate it.
5. `pip install -r requirements.txt`
6. `python training.py`   (trains + saves model_dir/house_price_model.joblib)
7. `uvicorn main:app --reload`   (after main.py is built)

## Architecture
The React frontend never loads the model. It calls the FastAPI service over HTTP.
The saved artifact is a full Pipeline (preprocessing + model), so the API sends
raw human-readable values and the pipeline handles imputation, scaling, and
one-hot encoding internally.
