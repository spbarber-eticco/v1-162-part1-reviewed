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
device = "mps" if torch.backends.mps.is_built() else "cpu"
logger.info(f"Dispositivo: {device}")

# Configurar el modelo de embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model,
    model_kwargs={"device": device},
    encode_kwargs = {'normalize_embeddings': False}
)

tokenizer = AutoTokenizer.from_pretrained(llm_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 2048

model = AutoModelForCausalLM.from_pretrained(
    llm_model, 
    max_position_embeddings=2048,
    torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
).to(device)

model.config.max_length = 2048
model.config.max_position_embeddings = 2048
model.config.pad_token_id = tokenizer.eos_token_id

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
    model_kwargs={"max_length": 2048, "max_new_tokens": 512}
)

def check_collection():
    try:
        count = vector_store._embedding_length
        logger.info(f"Número de documentos en la colección: {count}")
    except Exception as e:
        logger.error(f"Error al verificar la colección: {str(e)}")

def count_documents():
    try:
        with vector_store.connection.connect() as conn:
            result = conn.execute(f"SELECT COUNT(*) FROM {vector_store.collection_name}").fetchone()
            logger.info(f"Número real de documentos en la colección: {result[0]}")
    except Exception as e:
        logger.error(f"Error al contar documentos: {str(e)}")


try:
    # Configuración de la base de datos
    connection_string = "postgresql+psycopg://spbarber@localhost:5432/erpbrain_rag"
    collection_name = "collection_test"
    # Inicializar el almacén de vectores
    vector_store = PGVector(
        collection_name=collection_name,
        connection=connection_string,
        embeddings=embeddings,
        distance_strategy=DistanceStrategy.COSINE,
        pre_delete_collection=False,  # Ya no necesitamos esto porque manejamos la eliminación manualmente
        embedding_length=768,  # Longitud correcta para all-mpnet-base-v2
        use_jsonb=True  # Usar JSONB para metadatos
    )
    logger.info("Vector store inicializado correctamente")

    # Llama a esta función al inicio de tu script
    check_collection()

    # Llama a esta función después de inicializar vector_store
    count_documents()
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

# Inicializar el modelo de lenguaje
# try:
#     llm = ChatOpenAI(temperature=0, model='gpt-4o-mini', streaming=True)
#     logger.info("Modelo de lenguaje inicializado")
# except Exception as e:
#     logger.error(f"Error al inicializar el modelo de lenguaje: {e}")
#     raise

class RagInput(TypedDict):
    question: str

def retrieve_and_log(question):
    logger.info(f"Recuperando contexto para la pregunta: {question}")
    try:
        results = vector_store.similarity_search(question, k=3)
        logger.info(f"Resultados obtenidos: {len(results)}")
        
        max_context_length = 1000  # Ajusta este valor según sea necesario
        contexts = []
        total_length = 0
        for i, doc in enumerate(results):
            if total_length + len(doc.page_content) > max_context_length:
                break
            contexts.append(doc.page_content)
            total_length += len(doc.page_content)
            logger.info(f"Documento {i+1} - Longitud: {len(doc.page_content)}")
            logger.info(f"Primeros 100 caracteres: {doc.page_content[:100]}")
        
        joined_context = " ".join(contexts)
        logger.info(f"Longitud total del contexto: {len(joined_context)} caracteres")
        return joined_context
    except Exception as e:
        logger.error(f"Error al recuperar el contexto: {str(e)}")
        return ""

def log_final_response(response):
    logger.info(f"Respuesta generada: {response[:50]}...")
    return response

# Construir la cadena final
final_chain = (
    RunnableParallel(
        {
            "context": lambda x: retrieve_and_log(x["question"]),
            "question": itemgetter("question")
        }
    )
    | ANSWER_PROMPT
    | llm
    | StrOutputParser()
).with_types(input_type=RagInput)

logger.info("Cadena final construida y lista para su uso")

def log_chain_execution(input_data):
    logger.info(f"Pregunta recibida: {input_data['question']}")
    try:
        context = retrieve_and_log(input_data['question'])
        full_prompt = f"Answer given the following context:\n{context}\n\nQuestion: {input_data['question']}"
        
        # Limitar la longitud del prompt
        max_tokens = 2000  # Ajusta este valor según sea necesario
        encoded_prompt = tokenizer.encode(full_prompt, truncation=True, max_length=max_tokens)
        truncated_prompt = tokenizer.decode(encoded_prompt)
        
        logger.info(f"Longitud del prompt en tokens: {len(encoded_prompt)}")
        
        result = llm.invoke(truncated_prompt)
        logger.info(f"Respuesta generada: {result[:100]}...")
        return result
    except Exception as e:
        logger.error(f"Error en log_chain_execution: {str(e)}")
        return f"Lo siento, hubo un error al procesar tu pregunta: {str(e)}"

logger.info("Cadena final construida y lista para su uso")