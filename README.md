
---

# Lung Cancer Prediction API with Makefile Automation (FastAPI Version)

This project provides an automated pipeline for training and deploying a lung cancer prediction model using **Gradient Boosting** and **FastAPI**. A Makefile is included to streamline environment setup, model training, prediction, and cleanup.

---

## Features

- **Makefile Automation**  
  Simplifies setup, training, prediction, and cleanup through terminal commands.

- **Gradient Boosting Classifier**  
  A robust and accurate model for lung cancer classification.

- **FastAPI REST API**  
  Lightweight, high-performance backend for model interaction.

- **Data Preprocessing**  
  Includes cleaning, feature extraction, and class imbalance correction.

---

## Makefile Commands (Windows Instructions)

Ensure you are using a terminal with `make` support (e.g., Git Bash).

### 1. Setup Environment

```bash
make init
```

Creates a virtual environment, upgrades `pip`, and installs required dependencies from `requirements.txt`.

### 2. Run the API Server

```bash
make run
```

Starts the FastAPI development server using Uvicorn.

### 3. Train the Model

```bash
make train
```

This command sends a `POST` request to the `/api/train` endpoint.

Optional arguments can be set by modifying the `Makefile`:
- `DATA_PATH`: Path to the dataset CSV file
- `MODEL_PATH`: Output path for the trained model

### 4. Make a Prediction

```bash
make predict
```

This sends a `POST` request to the `/api/predict` endpoint using default or customized features.

To use custom features, modify the `FEATURES` variable in the Makefile:
```bash
FEATURES = "[55, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1]"
```

### 5. Clean Environment

```bash
make clean
```

Removes:
- `venv/`
- `__pycache__/`

---

## API Endpoints

### 1. Train Model

- **Endpoint**: `/api/train`  
- **Method**: `POST`  
- **Request Body**:
```json
{
  "data_path": "data/survey_lung_cancer.csv",
  "model_save_path": "models/best_gboost_model.pkl"
}
```

- **Response**:
```json
{
  "message": "Model trained successfully",
  "result": {
    "accuracy": 0.9375,
    "best_params": {
      "learning_rate": 0.01,
      "max_depth": 3,
      "n_estimators": 200
    }
  }
}
```

### 2. Make Prediction

- **Endpoint**: `/api/predict`  
- **Method**: `POST`  
- **Request Body**:
```json
{
  "model_path": "models/best_gboost_model.pkl",
  "features": [60, 2, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2]
}
```

- **Response**:
```json
{
  "prediction": [1],
  "result_label": "Present"
}
```

---

## Dataset Format

Your CSV dataset must include the following columns:

- `AGE`  
- `GENDER`  
- `SMOKING`  
- `ANXIETY`  
- `YELLOW_FINGERS`  
- `PEER_PRESSURE`  
- `CHRONIC DISEASES`  
- `FATIGUE`  
- `ALLERGY`  
- `WHEEZING`  
- `ALCOHOL CONSUMPTION`  
- `COUGHING`  
- `LUNG_CANCER` (Target: 0 = Absent, 1 = Present)

---

## Manual Setup (Without Makefile)

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment (Windows):

```bash
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the server:

```bash
uvicorn run:app --reload
```

---

## Project Structure

```
project/
├── app/
│   ├── __init__.py
│   ├── models.py
│   ├── routes.py
├── data/
│   └── survey_lung_cancer.csv
├── models/
│   ├── best_gboost_model.pkl
├── requirements.txt
├── run.py
├── Makefile
├── README.md
```

---

## Author

- Tasnim Mtir

---

## Acknowledgments

This project leverages the following technologies:

- [Scikit-learn](https://scikit-learn.org/)
- [Imbalanced-learn](https://imbalanced-learn.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Joblib](https://joblib.readthedocs.io/)
- [Make](https://www.gnu.org/software/make/)

---

