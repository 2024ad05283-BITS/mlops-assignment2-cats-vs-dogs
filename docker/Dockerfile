FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# CPU-only PyTorch
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]