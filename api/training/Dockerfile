FROM python:3.10-slim

WORKDIR /app

RUN pip install fastapi uvicorn boto3 numpy tensorflow scikit-learn

COPY ./api/training/ /app

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]