FROM python:3.10-slim

WORKDIR /app

# system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# copy requirements
COPY requirements.txt .

# install python packages
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# expose port
EXPOSE 5000

# run server
CMD ["python", "app.py"]