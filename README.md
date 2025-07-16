# Space Image Classifier

A satellite image classifier with REST API and AI model trained on EuroSAT dataset. Supports land cover classification (forest, water, crop, etc.).

## Features

- Multi-class satellite image classification (10 classes from EuroSAT)
- REST API built with FastAPI
- Rate limiting to protect endpoints
- Logging and experiment tracking with Weights & Biases

## Quickstart

**Install Dependencies**

```bash
    git clone https://github.com/papakapa/space-img-cls.git
    cd space-img-cls
    cp .env.example .env
    make install
    wandb login
```

**Train the model**

```bash
  python models/train.py
```


**Run API**
```bash
  make dev
```