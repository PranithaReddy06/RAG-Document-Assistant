from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model=settings.embedding_model)

vectorstore = Chroma(collection_name="documents", embedding_function=embeddings, persist_directory="./chroma_db")

if __name__ == "__main__":
    duplicate = vectorstore.get(
        where={
            "source": r"documents/uploads\0557f799-48c5-48a0-9a26-8eeab0fc108a_ecourt.pdf"
        },
        include=["metadatas"]
    )

    ids = duplicate["ids"]

    if ids:
        vectorstore.delete(ids=ids)
        print(f"Deleted {len(ids)} duplicate chunks.")
    else:
        print("No duplicate chunks found.")