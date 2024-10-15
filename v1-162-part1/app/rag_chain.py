import os
import logging
import torch
from operator import itemgetter
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.runnables import RunnableParallel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


os.environ["TOKENIZERS_PARALLELISM"] = "false"
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()
logger.info("Variables de entorno cargadas")

# Definir modelos
llm_model = "meta-llama/Llama-3.2-3B-Instruct"
embedding_model = "sentence-transformers/all-mpnet-base-v2"

# Configurar el dispositivo
# device = "mps" if torch.backends.mps.is_built() else "cpu"
device = "cpu"
logger.info(f"Dispositivo: {device}")

# Configurar el modelo de embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model,
    model_kwargs={"device": device},
    encode_kwargs = {'normalize_embeddings': False}
)

tokenizer = AutoTokenizer.from_pretrained(
    llm_model,
    max_position_embeddings=2048,
    torch_dtype=torch.float16 if torch.backends.mps.is_built() else torch.float32
    )
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.model_max_length = 2048

# Imprimir pad y eos tokens
logger.info(f"Pad token: {tokenizer.pad_token}")
logger.info(f"Eos token: {tokenizer.eos_token}")

model = AutoModelForCausalLM.from_pretrained(
    llm_model, 
    max_position_embeddings=2048,
    torch_dtype=torch.float16 if torch.backends.mps.is_built() else torch.float32
).to(device)

model.config.max_length = 2048
model.config.max_position_embeddings = 2048
model.generation_config.max_length = 2048
model.generation_config.max_new_tokens = 512

pipe = pipeline(
    "text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=512,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.2,
    device=device,
    max_length=2048,
)

llm = HuggingFacePipeline(
    pipeline=pipe,
    model_kwargs={"max_length": 2048, "max_new_tokens": 512},
    callbacks=None
)
def check_collection():
    try:
        count = vector_store._embedding_length
        logger.info(f"Número de documentos en la colección: {count}")
    except Exception as e:
        logger.error(f"Error al verificar la colección: {str(e)}")


try:
    # Configuración de la base de datos
    connection_string = "postgresql+psycopg://spbarber@localhost:5432/erpbrain_rag"
    collection_name = "collection_test"

    vector_store = PGVector(
        collection_name=collection_name,
        connection=connection_string,
        embeddings=embeddings,
        distance_strategy=DistanceStrategy.COSINE,
        pre_delete_collection=False,
        embedding_length=768,
        use_jsonb=True
    )

    logger.info("Vector store inicializado correctamente")

    # Llama a esta función al inicio de tu script
    check_collection()
except Exception as e:
    logger.error(f"Error al inicializar el vector store: {e}")
    raise

template = """
Answer given the following context:
{context}

Question: {question}
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(template)
logger.info("Prompt de respuesta creado")


class RagInput(TypedDict):
    question: str


def log_final_response(response):
    logger.info(f"Respuesta generada: {response[:50]}...")
    return response

# Construir la cadena final
logger.info("Construyendo cadena final...")

final_chain = (
    RunnableParallel(
        {
            "context": lambda x: vector_store.similarity_search(x["question"], k=3),
            "question": itemgetter("question")
        }
    )
    | ANSWER_PROMPT
    | llm
    | StrOutputParser()
).with_types(input_type=RagInput)

logger.info("Cadena final construida y lista para su uso")