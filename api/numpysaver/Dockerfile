FROM python:3.10-slim

WORKDIR /app

COPY ./api/numpysaver/ /app

RUN pip install fastapi uvicorn boto3 numpy

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]