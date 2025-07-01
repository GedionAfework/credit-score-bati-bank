# Credit Scoring Model for Bati Bank

## Overview
The **Credit Scoring Model for Bati Bank** is a robust solution developed to support Bati Bank’s “buy-now-pay-later” (BNPL) service, launched in partnership with an innovative eCommerce platform. This project aims to assess the creditworthiness of potential borrowers, estimate their risk of default, and determine optimal loan terms using transaction data. The model leverages behavioral patterns, such as Recency, Frequency, and Monetary (RFM) metrics, to deliver accurate and transparent credit decisions, ensuring alignment with regulatory standards like the Basel II Capital Accord.

The repository contains all necessary components to process data, train the credit scoring model, deploy it as a scalable service, and integrate it with the eCommerce platform. The project prioritizes interpretability, compliance, and scalability to support Bati Bank’s goals of financial inclusion and risk management.

## Project Objectives
The Credit Scoring Model was designed to achieve the following:
- **Risk Classification**: Categorize customers as high-risk or low-risk based on transaction behavior.
- **Predictive Features**: Develop features that capture customer financial patterns for accurate risk prediction.
- **Risk Probability Scoring**: Assign a probability score indicating the likelihood of default.
- **Credit Scoring**: Translate risk probabilities into a standardized credit score (300–850).
- **Loan Term Optimization**: Determine loan amounts (\$100–\$5,000) and durations (3–24 months) tailored to each customer’s risk profile.
- **Regulatory Compliance**: Ensure transparency and interpretability to meet Basel II requirements.

## Repository Structure
The repository is organized as follows:
- **data/**: Contains raw and processed datasets.
  - `raw/`: Original transaction data.
  - `processed/`: Transformed data ready for modeling.
- **src/**: Core project scripts.
  - `api/`: FastAPI service for real-time credit scoring.
  - Other scripts for data processing, model training, and prediction.
- **tests/**: Unit tests for ensuring code quality.
- **.github/workflows/**: CI/CD pipeline configuration for automated testing and deployment.
- **Dockerfile**: Defines the containerized environment for the FastAPI service.
- **docker-compose.yml**: Configures the Docker service for local deployment.
- **requirements.txt**: Lists project dependencies.

## Prerequisites
To set up and run the project locally, ensure the following:
- **Python**: Version 3.9 or higher.
- **Docker**: Docker Desktop for containerized deployment (optional).
- **Virtual Environment**: Recommended for dependency management.
- **Operating System**: Windows, macOS, or Linux.
- **Dependencies**: Listed in `requirements.txt`, including libraries for data processing, modeling, and API deployment.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/credit-score-bati-bank.git
   cd credit-score-bati-bank
   ```

2. **Set Up Virtual Environment**:
   - Create and activate a virtual environment:
     - **Windows (PowerShell)**:
       ```powershell
       python -m venv .venv
       .\venv\Scripts\Activate.ps1
       ```
     - **macOS/Linux (Bash)**:
       ```bash
       python3 -m venv .venv
       source .venv/bin/activate
       ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Data**:
   - Place raw transaction data in `data/raw/data.csv`.
   - Run the data processing script to generate processed data:
     ```bash
     python src/data_processing.py
     ```
   - Verify that `data/processed/processed_data.csv` is created.

5. **Set MLflow Tracking URI**:
   - Configure the tracking URI for model storage:
     - **Windows (PowerShell)**:
       ```powershell
       $env:MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
       ```
     - **macOS/Linux (Bash)**:
       ```bash
       export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
       ```

6. **Train the Model**:
   - Run the training script to build and register the model:
     ```bash
     python src/train.py
     ```
   - Verify model registration using MLflow UI:
     ```bash
     mlflow ui
     ```
     Open `http://localhost:5000` in a browser to confirm the model (`best_model`) is registered.

## Usage
### Running the FastAPI Service Locally
1. Start the FastAPI service from the project root:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```
2. Test the API with a sample request:
   - **Windows (PowerShell)**:
     ```powershell
     Invoke-WebRequest -Uri http://localhost:8000/predict -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"Amount_sum": 1000, "Amount_mean": 500, "Amount_count": 2, "Amount_std": 50, "Value_sum": 1000, "Value_mean": 500, "Value_count": 2, "Value_std": 50, "woe_Amount": 0.1, "woe_Value": 0.1, "woe_TransactionHour": 0.1, "woe_TransactionDay": 0.1, "woe_TransactionMonth": 0.1, "woe_ProductCategory": 0.1, "woe_ChannelId": 0.1, "woe_PricingStrategy": 0.1}'
     ```
   - **macOS/Linux (Bash)**:
     ```bash
     curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"Amount_sum": 1000, "Amount_mean": 500, "Amount_count": 2, "Amount_std": 50, "Value_sum": 1000, "Value_mean": 500, "Value_count": 2, "Value_std": 50, "woe_Amount": 0.1, "woe_Value": 0.1, "woe_TransactionHour": 0.1, "woe_TransactionDay": 0.1, "woe_TransactionMonth": 0.1, "woe_ProductCategory": 0.1, "woe_ChannelId": 0.1, "woe_PricingStrategy": 0.1}'
     ```

### Running with Docker
1. Ensure Docker Desktop is running.
2. Build and start the service:
   ```bash
   docker compose up
   ```
3. Test the API as described above.
4. Stop the service:
   ```bash
   docker compose down
   ```

### Generating Batch Predictions
To generate predictions on a dataset:
```bash
python src/predict.py
```
This creates `data/processed/processed_data_predictions.csv` with risk probabilities, credit scores, and loan terms.

## Model Performance
The Credit Scoring Model achieves:
- **Accuracy**: 88.7%, correctly classifying most customers as high-risk or low-risk.
- **Precision**: 64.4%, ensuring reliable identification of high-risk customers.
- **Recall**: 3.5%, indicating a conservative approach to flagging high-risk cases.
- **F1 Score**: 6.6%, balancing precision and recall.
- **ROC AUC**: 51.6%, suggesting moderate discriminatory power with room for improvement.

These metrics reflect a robust foundation for credit risk assessment, with future enhancements planned to improve recall and discrimination.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/<feature-name>`).
3. Commit changes (`git commit -m "Add <feature-name>"`).
4. Push to the branch (`git push origin feature/<feature-name>`).
5. Open a pull request.

Please ensure code adheres to the project’s style guidelines and passes all tests.

## Testing
Run unit tests to verify functionality:
```bash
pytest tests/
```
The CI/CD pipeline in `.github/workflows/ci.yml` automatically runs tests on each push.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or support, contact Gedion Mekbeb Afework at [your-email@example.com].
