from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from app.rag_chain import final_chain
import logging


logger = logging.getLogger(__name__)

# Configurar FastAPI
app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# Añadir rutas para la cadena RAG
add_routes(app, final_chain, path="/rag")

logger.info("Rutas de FastAPI configuradas")

if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando servidor FastAPI")
    uvicorn.run(app, host="0.0.0.0", port=8000)
