FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]