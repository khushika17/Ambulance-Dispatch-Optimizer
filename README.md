# Ambulance Dispatch Optimizer

ML-driven emergency dispatch decision support with a web UI.

## Project Layout

```
ambulance-dispatch-optimizer-main/
├── backend/
│   ├── app.py                  # Flask API & static-file server
│   ├── train.py                # End-to-end ML pipeline
│   ├── src/
│   │   ├── config.py           # Paths
│   │   ├── data_generation.py  # Synthetic data
│   │   ├── preprocessing.py    # Feature engineering & scaling
│   │   ├── model_training.py   # Train/evaluate multiple models
│   │   ├── prediction_service.py
│   │   └── visualization.py    # Plot generation
│   ├── data/                   # CSV dataset
│   ├── models/                 # Saved .pkl + metrics.json
│   └── static/images/          # Generated charts
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── requirements.txt
└── run.py
```

## Quick start

```powershell
pip install -r requirements.txt
python run.py all
```

Then open http://127.0.0.1:5000 in a browser.

### Or step by step

```powershell
python backend/train.py     # train the model
python backend/app.py       # start the web app
```

## API

- `GET  /api/health` - service status
- `GET  /api/model-info` - metrics and visualizations
- `POST /api/predict` - dispatch prediction (JSON body)
- `POST /api/retrain` - retrain the pipeline (`{"n_samples": 5000}`)
