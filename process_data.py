from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings, Document
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
import chromadb
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables")

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Fixed chunk parameters
CHUNK_SIZE = 356
CHUNK_OVERLAP = 50



class DocumentProcessor:
    def __init__(self, db_path: str = "chroma_db", collection_name: str = "documents"):
        """Initialize document processor with ChromaDB"""
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.text_splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self._setup_database()

    def _setup_database(self):
        """Set up ChromaDB with basic error handling"""
        try:
            self.db_path.mkdir(parents=True, exist_ok=True)
            self.db = chromadb.PersistentClient(path=str(self.db_path))
            self.chroma_collection = self.db.get_or_create_collection(self.collection_name)
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            logger.info(f"Initialized ChromaDB at {self.db_path}")
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {str(e)}")
            raise

    def process_documents(self, data_dir: str):
        """Process documents with simple sentence splitting"""
        try:
            # Load documents
            reader = SimpleDirectoryReader(
                data_dir,
                recursive=True,
                filename_as_id=True
            )
            documents = reader.load_data()
            logger.info(f"Loaded {len(documents)} documents from {data_dir}")

            # Create index with sentence splitter
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                transformations=[self.text_splitter],
                show_progress=True
            )
            logger.info("Successfully processed and indexed documents")
            return index
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise


def main():
    processor = DocumentProcessor()
    processor.process_documents("data")


if __name__ == "__main__":
    main()