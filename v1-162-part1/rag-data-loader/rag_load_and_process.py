import os
import torch
import uuid

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from langchain_experimental.text_splitter import SemanticChunker
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine, text
from sqlalchemy_utils import database_exists, create_database
from langchain_core.documents import Document

load_dotenv()

# Definir modelos
embedding_model = "sentence-transformers/all-mpnet-base-v2"

# Configurar el dispositivo
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Configurar el modelo de embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model,
    model_kwargs={"device": device},
    encode_kwargs = {'normalize_embeddings': False}
)


loader = DirectoryLoader(
    os.path.abspath("../pdf-documents"),
    glob="**/*.pdf",
    use_multithreading=True,
    show_progress=True,
    max_concurrency=50,
    loader_cls=UnstructuredPDFLoader,
)
docs = loader.load()
# print(f"Documentos cargados: {len(docs)}")
# print(f"page_content: {docs[0].page_content}")
# print(f"metadata: {docs[0].metadata}")

text_splitter = SemanticChunker(
    embeddings=embeddings,
    # Cortar los textos cada salto de linea
    sentence_split_regex = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s",
)
print(f"Text splitter: {text_splitter}")
print(f"page_content: {docs[0].page_content}")

# docs = text_splitter.create_documents(docs)
# print(f"Documentos procesados: {len(docs)}")
# print(f"page_content: {docs[0].page_content}")

# flattened_docs = [doc[0] for doc in docs if doc]
# chunks = text_splitter.split_documents(flattened_docs)
# print(f"Documentos procesados: {len(chunks)}")


# Configuración de la base de datos
connection_string = "postgresql+psycopg://spbarber@localhost:5432/erpbrain_rag"
collection_name = "collection_test"

# Asegurarse de que la base de datos existe
engine = create_engine(connection_string)
if not database_exists(engine.url):
    create_database(engine.url)

def delete_collection():
    try:
        with engine.connect() as connection:
            # Eliminar la tabla de la colección
            connection.execute(text(f"DROP TABLE IF EXISTS {collection_name}"))
            # Eliminar la entrada en langchain_pg_collection
            connection.execute(text(f"DELETE FROM langchain_pg_collection WHERE name = '{collection_name}'"))
            connection.commit()
        print(f"Colección '{collection_name}' eliminada con éxito.")
    except Exception as e:
        print(f"Error al eliminar la colección: {e}")

def create_collection_table():
    try:
        with engine.connect() as connection:
            # Crear la tabla de la colección con la estructura correcta
            connection.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {collection_name} (
                    id UUID PRIMARY KEY,
                    collection_id UUID,
                    embedding FLOAT[],
                    document TEXT,
                    cmetadata JSONB
                );
            """))
            print(f"Tabla '{collection_name}' creada o ya existe.")
    except Exception as e:
        print(f"Error al crear la tabla: {e}")

def recreate_embeddings(force_recreate=False):
    try:
        if force_recreate:
            delete_collection()
        
        # Crear la tabla si no existe
        create_collection_table()

        # Inicializar PGVector con JSONB para metadatos
        vector_store = PGVector(
            collection_name=collection_name,
            connection=connection_string,
            embeddings=embeddings,
            distance_strategy=DistanceStrategy.COSINE,
            pre_delete_collection=True,  # Ya no necesitamos esto porque manejamos la eliminación manualmente
            embedding_length=768,  # Longitud correcta para all-mpnet-base-v2
            use_jsonb=True  # Usar JSONB para metadatos
        )

        print("Control vector_store:", vector_store.connection_string_from_db_params)
        print("collection_name =", vector_store.collection_name)

                # Verificar si ya existen embeddings
        if vector_store.collection_name == 'collection_test':
            print("Recreando embeddings...")

            # Cargar los documentos en la base de datos
            vector_store.add_documents(documents=docs, ids=[str(uuid.uuid4()) for _ in docs])
            print(f"Documentos procesados: {len(docs)}")
        else:
            print("Embeddings ya existen.")
    except Exception as e:
        print(f"Error al recrear embeddings: {e}")
        
recreate_embeddings(force_recreate=True)
    #     # Verificar si ya existen embeddings
    #     if not force_recreate:
    #         try:
    #             vector_store.similarity_search("test", k=1)
    #             print("Los embeddings ya existen. No se recrearán.")
    #             return
    #         except Exception:
    #             print("No se encontraron embeddings existentes. Creando nuevos...")
        
    #     # Si llegamos aquí, recreamos los embeddings
    #     print("Recreando embeddings...")
        
    #     # Crear nueva colección
    #     text_splitter = SemanticChunker(embeddings=embeddings)
    #     # flattened_docs = [doc[0] for doc in docs if doc]
    #     chunks = text_splitter.split_documents(docs)
    #     print(f"Documentos procesados: {len(chunks)}")
    #     ids = [str(uuid.uuid4()) for _ in chunks]

    #     # for id, chunk in zip(ids, chunks):
    #     #     print(f"Creando chunk {id}... con chunk {chunk}")
    #     # vector_store.add_documents(chunks, ids=ids)
    #     # vector_store.add_documents(docs, ids=[doc.id for doc in docs])
    #     vector_store.add_documents(chunks, ids=ids)
    #     print("Embeddings recreados exitosamente.")
    # except Exception as e:
    #     print(f"Error al recrear embeddings: {e}")

# Llamar a la función para recrear embeddings
# Cambia False a True si quieres forzar la recreación y eliminar la colección existente

# embeddings = OpenAIEmbeddings(model='text-embedding-3-small', )

# try:
#     # Intenta crear la colección
#     PGVector.create_collection(embedding=embeddings, connection_string=connection_string)
#     print(f"Colección creada con éxito.")
# except Exception as e:
#     print(f"Error al crear la colección: {e}")
#     print("Continuando con la carga de documentos...")


# text_splitter = SemanticChunker(
#     embeddings=embeddings
# )

# flattened_docs = [doc[0] for doc in docs if doc]
# chunks = text_splitter.split_documents(flattened_docs)

# try:
#     # Carga los documentos en la base de datos
#     PGVector.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         collection_name=collection_name,
#         connection_string=connection_string,
#         pre_delete_collection=True,
#     )
#     print("Documentos cargados con éxito en la base de datos.")
# except Exception as e:
#     print(f"Error al cargar los documentos: {e}")