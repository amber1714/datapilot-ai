# DataPilot AI

Autonomous Data Science Assistant.

<<<<<<< HEAD
## Step 1 Complete

The Streamlit app can now:

- Upload CSV files
- Preview the dataset
- Show rows and columns
- Display data types
- Count missing values
- Detect duplicate rows
- Show numerical statistics
- Show categorical statistics
- Display a dataset summary

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

A sample dataset is available at:

`sample_data/sample_students.csv`

## Next Step

Step 2 will add target-column selection and automatic classification/regression detection.
=======
## Step 1
- Upload and inspect CSV datasets

## Step 2
- Select a target column
- Automatically detect classification or regression

## Step 3
- Automatic preprocessing
- Missing-value handling
- Scaling
- One-hot encoding
- Train/test split

## Step 4
- Train and compare multiple models
- Classification metrics
- Regression metrics

## Step 5
- Confusion matrix
- ROC curve for binary classification
- Feature importance
- Actual vs predicted plot for regression
- Residual plot for regression

## Run

```bash
py -m pip install -r requirements.txt
py -m streamlit run streamlit_app.py
```

## Next Step

Step 6 will add:
- Hyperparameter tuning with Optuna
- Better model-selection logic
- Downloadable results


## Step 6
- Hyperparameter tuning with Optuna
- Tune Random Forest and XGBoost models
- Adjustable Optuna trial count
- Display best hyperparameters
- Download model-comparison CSV
- Download HTML report

## Next Step
Step 7 will add model persistence, saved experiments, and prediction on new data.


## Step 7
- Save a trained model as a `.joblib` artifact
- Store preprocessing and target-encoding logic with the model
- Reload saved DataPilot models
- Enter new feature values through the Streamlit interface
- Generate predictions for classification or regression
- Show prediction confidence when probability estimates are available

## Next Step
Step 8 will add experiment history and model metadata storage.


## Step 7 Fix
- Trained models are stored in Streamlit session state.
- The prediction interface remains available after widget interactions.
- New predictions are submitted through a Streamlit form.


## Step 8
- SQLite experiment-history database
- Automatically save every completed model-training run
- Store dataset name, target, problem type, split, feature count and models
- Store the top-performing model and primary evaluation metric
- Store the full model-comparison metrics
- Browse previous experiments directly in Streamlit
- View detailed metadata for each experiment

The local SQLite file is created automatically as:

`datapilot.db`

It is excluded from Git so local experiment history is not committed accidentally.

## Next Step
Step 9 will introduce the FastAPI backend and move core ML actions behind API endpoints.


## Step 9
FastAPI backend added.

### API endpoints
- `GET /`
- `GET /health`
- `POST /analyze`
- `POST /detect-problem`
- `POST /train`

### Run the FastAPI backend

```bash
py -m uvicorn api.main:app --reload
```

Open:
- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`

### Run Streamlit separately

Open a second terminal:

```bash
py -m streamlit run streamlit_app.py
```

## Next Step
Step 10 will connect Streamlit to the FastAPI backend using HTTP requests.


## Step 10
Streamlit is now connected to FastAPI through HTTP requests.

### Architecture

```text
Browser
  |
  v
Streamlit :8501
  |
  | HTTP
  v
FastAPI :8000
  |
  v
ML pipeline
```

### Terminal 1 — FastAPI

```bash
py -m uvicorn api.main:app --reload
```

### Terminal 2 — Streamlit

```bash
py -m streamlit run streamlit_app.py
```

### Backend URL

By default the frontend uses:

`http://127.0.0.1:8000`

For deployment, set:

`DATAPILOT_API_URL`

to the public FastAPI URL.

## Next Step

Step 11 will Dockerize the FastAPI backend and prepare deployment configuration.


## Step 11 — Dockerized FastAPI Backend

The FastAPI backend can now run inside Docker.

### New files

- `Dockerfile`
- `.dockerignore`
- `requirements-api.txt`
- `docker-compose.yml`
- `railway.toml`

### Build the Docker image

```bash
docker build -t datapilot-api .
```

### Run the Docker container

```bash
docker run --rm -p 8000:8000 datapilot-api
```

Then open:

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

### Using Docker Compose

```bash
docker compose up --build
```

Stop it with:

```bash
docker compose down
```

### Railway preparation

`railway.toml` is included for deployment.

Railway should build the repository using the included Dockerfile and run:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

The health endpoint is:

`/health`

## Next Step

Step 12 will add GitHub Actions to automatically test the project on every push.


## Step 12 — GitHub Actions CI

Continuous Integration is now configured.

### Added files
- `.github/workflows/ci.yml`
- `requirements-test.txt`
- `tests/test_api.py`
- `tests/test_problem_detection.py`
- `tests/test_training.py`

### Run tests locally

```bash
py -m pip install -r requirements-test.txt
py -m pytest -q
```

### What GitHub Actions checks
On every push to `main` and every pull request targeting `main`, GitHub will:
1. Check out the repository
2. Install Python 3.12
3. Install test dependencies
4. Run the pytest suite

## Next Step
Step 13 will push the updated project to GitHub and deploy the FastAPI backend to Railway.
>>>>>>> 85d9c89 (Update DataPilot AI through Step 12)
