from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
import numpy as np

app = FastAPI(title="API Otimizada com Álgebra Linear")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

@app.post("/treinar-rapido")
def treinar_rapido(qtd_amostras: int = 1_000_000):
    inicio = time.time()
    
    np.random.seed(42)
    X = np.random.randn(qtd_amostras, 5)
    y = np.random.randint(0, 2, size=qtd_amostras)
    
    w = np.zeros(5)
    b = 0.0
    alpha = 0.01
    epocas = 5
    
    for epoch in range(epocas):
        z = np.dot(X, w) + b
        preds = sigmoid(z)
        
        error = preds - y
        dj_dw = np.dot(X.T, error) / qtd_amostras
        dj_db = np.sum(error) / qtd_amostras
        
        w -= alpha * dj_dw
        b -= alpha * dj_db
        
    tempo_decorrido = time.time() - inicio
    
    return {
        "metodo": "Otimizado (Álgebra Linear + NumPy Vetorizado)",
        "amostras_processadas": qtd_amostras,
        "tempo_segundos": round(tempo_decorrido, 4)
    }
