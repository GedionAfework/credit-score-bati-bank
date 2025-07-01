# Credit Risk Model for Bati Bank

## Overview

This repository contains the implementation of a Credit Scoring Model for Bati Bank in partnership with an eCommerce company to enable a buy-now-pay-later service. The model leverages behavioral data (Recency, Frequency, Monetary - RFM) to predict credit risk and assign credit scores.

## Project Structure

```
credit-score-bati-bank/
├── .github/workflows/ci.yml   # CI/CD configuration
├── data/                      # Data storage (ignored by .gitignore)
│   ├── raw/                  # Raw data
│   └── processed/            # Processed data
├── notebooks/
│   └── eda.ipynb             # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # Feature engineering script
│   ├── train.py             # Model training script
│   ├── predict.py           # Inference script
│   └── api/
│       ├── main.py          # FastAPI application
│       └── pydantic_models.py # Pydantic models
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
├── venv/                     # Virtual environment
└── README.md
```

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/GedionAfework/credit-score-bati-bank.git
   cd credit-score-bati-bank
   ```
2. **Set Up Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unix/Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Verify Setup:**
   Run a test script or notebook to ensure dependencies are installed correctly.

## Usage

- Place raw data in `data/raw/`.
- Run the EDA notebook: `jupyter notebook notebooks/eda.ipynb`.
- Execute feature engineering: `python src/data_processing.py`.

## Credit Scoring Business Understanding

(To be completed in Task 1)
