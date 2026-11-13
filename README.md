# AQI Prediction – Flask + Scikit-learn

A runnable, end-to-end educational regression project based on the structure of the supplied AzureDeployment project. It creates a sample dataset, trains a Random Forest pipeline, saves it with Joblib, and serves predictions through a Flask webpage and JSON API.

**Important:** `generate_dataset.py` creates synthetic teaching data. The model's predictions are only for demonstrating the ML deployment workflow; they are not official or scientifically valid AQI estimates.

## 1. Requirements

- Windows 10/11
- Python 3.11 or 3.12 (64-bit)
- VS Code (optional)

Check Python:

```powershell
py -3.11 --version
# or
py -3.12 --version
```

## 2. Create and activate a virtual environment

Open PowerShell in this project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this for the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

For Python 3.12, replace `-3.11` with `-3.12`.

## 3. Install libraries

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Generate the sample dataset

```powershell
python generate_dataset.py
```

Creates `data/aqi_sample.csv`.

## 5. Train and save the model

```powershell
python train_model.py
```

Prints MAE, RMSE and R², then creates `models/aqi_model.joblib`.

## 6. Start Flask

```powershell
python app.py
```

Open: http://127.0.0.1:5000

## 7. Test the JSON API

In a second PowerShell window:

```powershell
$body = @{
  max_wind_speed = 25
  max_temperature = 38
  min_temperature = 25
  average_temperature = 31
  average_wind_speed = 12
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict_api" `
  -Method Post -ContentType "application/json" -Body $body
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
```

## 8. Project structure

```text
aqi_prediction_flask/
├── app.py
├── generate_dataset.py
├── train_model.py
├── requirements.txt
├── Procfile
├── .python-version
├── README.md
├── data/
│   └── (aqi_sample.csv is generated)
├── models/
│   └── (aqi_model.joblib is generated)
├── templates/
│   └── home.html
└── static/
    └── style.css
```

## 9. Azure App Service deployment

1. Create a Python Web App using Python 3.11 or 3.12 on Linux.
2. Ensure the dataset and trained model are included in the deployed project. Run the two generation/training commands before deployment, or train in a build step.
3. Configure the startup command as:

```bash
gunicorn --bind 0.0.0.0:$PORT app:app
```

4. Deploy the project folder using your chosen deployment method.
5. Open the Web App URL and test `/health`.

The `Procfile` is included for platforms that use it. Azure deployment settings and subscription permissions are separate from the Python application.

## 10. API contract

`POST /predict_api` with `Content-Type: application/json`:

```json
{
  "max_wind_speed": 25,
  "max_temperature": 38,
  "min_temperature": 25,
  "average_temperature": 31,
  "average_wind_speed": 12
}
```

Example response:

```json
{
  "predicted_aqi": 200.0,
  "unit": "AQI"
}
```

The exact predicted value depends on the trained model and generated dataset.
