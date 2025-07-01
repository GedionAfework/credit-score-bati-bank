Credit Risk Model for Bati Bank
Overview
This repository contains the implementation of a Credit Scoring Model for Bati Bank in partnership with an eCommerce company to enable a buy-now-pay-later service. The model leverages behavioral data (Recency, Frequency, Monetary - RFM) to predict credit risk and assign credit scores.
Project Structure
credit-score-bati-bank/
├── .github/workflows/ci.yml # CI/CD configuration
├── data/ # Data storage (ignored by .gitignore)
│ ├── raw/ # Raw data
│ └── processed/ # Processed data
├── notebooks/
│ └── eda.ipynb # Exploratory Data Analysis
├── src/
│ ├── **init**.py
│ ├── data_processing.py # Feature engineering script
│ ├── train.py # Model training script
│ ├── predict.py # Inference script
│ └── api/
│ ├── main.py # FastAPI application
│ └── pydantic_models.py # Pydantic models
├── tests/
│ └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
├── .venv/ # Virtual environment
└── README.md

Setup Instructions

Clone the Repository:git clone https://github.com/GedionAfework/credit-score-bati-bank.git
cd credit-score-bati-bank

Set Up Virtual Environment:python -m venv .venv
source .venv/bin/activate # Unix/Linux/Mac

# or

.venv\Scripts\activate # Windows

Install Dependencies:pip install -r requirements.txt

Verify Setup:Run a test script or notebook to ensure dependencies are installed correctly.

Credit Scoring Business Understanding
How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
The Basel II Capital Accord emphasizes robust risk measurement and management to ensure financial stability. It requires banks to maintain adequate capital reserves based on credit risk assessments, mandating transparent and auditable models. An interpretable model, such as Logistic Regression with Weight of Evidence (WoE), facilitates compliance by allowing regulators and stakeholders to understand how risk scores are derived. Well-documented models ensure traceability and reproducibility, critical for meeting Basel II’s requirements for risk-weighted asset calculations and stress testing.
Why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?
Since the dataset lacks a direct "default" label, a proxy variable (e.g., based on RFM patterns or FraudResult) is necessary to categorize customers as high or low risk. This proxy serves as a substitute for actual default outcomes, enabling model training. However, relying on a proxy introduces risks, such as misclassification if the proxy poorly correlates with true default behavior. This could lead to approving high-risk customers (increasing defaults) or rejecting low-risk ones (losing revenue). Regular validation against real-world outcomes is essential to mitigate these risks.
What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?
Simple models like Logistic Regression with WoE are highly interpretable, aligning with Basel II’s transparency requirements, and are easier to explain to regulators and stakeholders. However, they may sacrifice predictive power for complex patterns. Complex models like Gradient Boosting offer higher accuracy by capturing non-linear relationships but are less interpretable, posing challenges for regulatory audits. In a regulated context, the trade-off favors interpretability to ensure compliance, though hybrid approaches (e.g., combining WoE with ensemble methods) can balance performance and transparency.
Usage

Place raw data in data/raw/.
Run the EDA notebook: jupyter notebook notebooks/eda.ipynb.
Execute feature engineering: python src/data_processing.py.
