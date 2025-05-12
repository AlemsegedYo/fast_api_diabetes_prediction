# main.py

from fastapi import FastAPI
from routers import predict_router

app = FastAPI()

app.include_router(predict_router.router)
