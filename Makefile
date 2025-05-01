# Variables
VENV = venv
PYTHON = $(VENV)/Scripts/python
PIP = $(VENV)/Scripts/pip
UVICORN = $(VENV)/Scripts/uvicorn

# Default paths
DATA_PATH = data/survey_lung_cancer.csv
MODEL_PATH = models/best_gboost_model.pkl
FEATURES_PATH = models/feature_names.pkl
FEATURES = [60, 2, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2]

.PHONY: init
init:
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

.PHONY: run
run:
	$(UVICORN) run:app --reload

.PHONY: train
train:
	curl -X POST http://127.0.0.1:8000/api/train \
	-H "Content-Type: application/json" \
	-d "{\"data_path\": \"$(DATA_PATH)\", \"model_save_path\": \"$(MODEL_PATH)\"}"

.PHONY: predict
predict:
	curl -X POST http://127.0.0.1:8000/api/predict \
	-H "Content-Type: application/json" \
	-d "{\"model_path\": \"$(MODEL_PATH)\", \"features\": $(FEATURES)}"

.PHONY: clean
clean:
	powershell -Command "if (Test-Path '__pycache__') { Remove-Item '__pycache__' -Recurse -Force }"
	powershell -Command "if (Test-Path 'venv') { Remove-Item 'venv' -Recurse -Force }"