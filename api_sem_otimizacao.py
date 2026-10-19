from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
import time
import numpy as np

app = FastAPI(title="API Sem Otimização")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/treinar-lento")
def treinar_lento(qtd_amostras: int = 20_000):
    inicio = time.time()
    
    np.random.seed(42)
    X = np.random.randn(qtd_amostras, 5)
    y = np.random.randint(0, 2, size=qtd_amostras)
    
    pesos = np.zeros(5)
    for epoch in range(5):
        for i in range(qtd_amostras):
            pred = 1.0 / (1.0 + np.exp(-np.dot(X[i], pesos)))
            erro = pred - y[i]
            pesos -= 0.01 * erro * X[i]
            
    tempo_decorrido = time.time() - inicio
    
    return {
        "metodo": "Sem otimização (Processamento iterativo/lento)",
        "amostras_processadas": qtd_amostras,
        "tempo_segundos": round(tempo_decorrido, 4)
    }
