import logging
import os
from langchain_community.document_loaders import DirectoryLoader
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

# load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables")

# Fixed chunk parameters
CHUNK_SIZE = 356
CHUNK_OVERLAP = 50


class DocumentProcessor:
    def __init__(self, persist_directory: str = "chroma_db", collection_name: str = "documents"):
        """Initialize document processor with ChromaDB"""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False
        )
        self._setup_database()

    def _setup_database(self):
        """Set up ChromaDB with basic error handling"""
        try:
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            logger.info(f"Initialized ChromaDB at {self.persist_directory}")
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {str(e)}")
            raise

    def process_documents(self, data_dir: str):
        """Process documents from a directory and store them in the vector database"""
        try:
            # Load documents from directory
            loader = DirectoryLoader(
                data_dir,
                recursive=True,
                show_progress=True
            )
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from {data_dir}")

            # Split documents into chunks
            splits = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(splits)} splits from documents")

            # Store documents in ChromaDB
            self.db.add_documents(splits)
            self.db.persist()
            logger.info("Successfully processed and indexed documents")

            return self.db
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise

def main():
    processor = DocumentProcessor()
    processor.process_documents("data")


if __name__ == "__main__":
    main()