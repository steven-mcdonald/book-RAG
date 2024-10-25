from langchain_community.vectorstores import Chroma
# from rag_app.get_embedding_function import get_embedding_function
# from image.src.rag_app.get_embedding_function import get_embedding_function

import shutil
import sys
import os
from langchain_community.vectorstores import Chroma
from populate_chroma_database import get_embedding_function
# if running as a quick test in local environment you can use the path below
# from get_embedding_function import get_embedding_function

CHROMA_PATH = os.environ.get("CHROMA_PATH", "chroma")
IS_USING_IMAGE_RUNTIME = bool(os.environ.get("IS_USING_IMAGE_RUNTIME", False))
CHROMA_DB_INSTANCE = None  # Reference to singleton instance of ChromaDB


def get_chroma_db():
    global CHROMA_DB_INSTANCE
    if not CHROMA_DB_INSTANCE:

        # Prepare the DB.
        CHROMA_DB_INSTANCE = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function(),
        )
        print(f"✅ Init ChromaDB {CHROMA_DB_INSTANCE} from {CHROMA_PATH}")

    return CHROMA_DB_INSTANCE