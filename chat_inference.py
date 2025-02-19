from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.memory import ChatMemoryBuffer
import chromadb
import os
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chat.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables")

# Configure LLM and embedding settings
Settings.llm = OpenAI(temperature=0.2, model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


class ChatBot:
    def __init__(self, db_path: str = "chroma_db", collection_name: str = "documents"):
        """Initialize chatbot with ChromaDB connection and memory"""
        self.db_path = db_path
        self.collection_name = collection_name
        self._setup_database()
        self._setup_memory()
        self._setup_query_engine()

    def _setup_database(self):
        """Set up ChromaDB connection"""
        try:
            self.db = chromadb.PersistentClient(path=self.db_path)
            self.chroma_collection = self.db.get_or_create_collection(self.collection_name)
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            logger.info(f"Connected to ChromaDB at {self.db_path}")
        except Exception as e:
            logger.error(f"ChromaDB connection failed: {str(e)}")
            raise

    def _setup_memory(self):
        """Set up chat memory"""
        try:
            self.memory = ChatMemoryBuffer.from_defaults(
                token_limit=1000
            )
            logger.info("Chat memory initialized")
        except Exception as e:
            logger.error(f"Memory setup failed: {str(e)}")
            raise

    def _setup_query_engine(self):
        """Set up the query engine with the vector store"""
        try:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=self.storage_context
            )

            self.query_engine = self.index.as_chat_engine(
                chat_memory=self.memory,
                streaming=True,
                similarity_top_k=3,
                system_prompt=(
                    "You are a helpful AI assistant. Use the provided context to answer "
                    "questions. If you're unsure or don't have enough context, say so. "
                    "Maintain a conversational tone and refer back to previous parts of "
                    "the conversation when relevant."
                )
            )
            logger.info("Query engine setup complete")
        except Exception as e:
            logger.error(f"Query engine setup failed: {str(e)}")
            raise

    def ask(self, question: str) -> str:
        """Process a question and return an answer"""
        try:
            logger.info(f"Received question: {question}")
            response = self.query_engine.chat(question)
            logger.info("Generated response successfully")
            return response
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

    def reset_memory(self):
        """Reset the chat memory"""
        try:
            self._setup_memory()
            self._setup_query_engine()
            logger.info("Chat memory reset successful")
            return "Memory has been reset. Starting a new conversation."
        except Exception as e:
            logger.error(f"Error resetting memory: {str(e)}")
            return f"Sorry, I encountered an error while resetting memory: {str(e)}"


def chat_loop():
    """Interactive chat loop"""
    chatbot = ChatBot()
    print("\nWelcome to the ChatBot! Type 'exit' to end the conversation or 'reset' to clear memory.\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['exit']:
                print("\nGoodbye!")
                break

            if user_input.lower() == 'reset':
                response = chatbot.reset_memory()
                print(f"\nBot: {response}")
                continue

            if not user_input:
                print("Please enter a question.")
                continue

            response = chatbot.ask(user_input)
            print(f"\nBot: {response}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {str(e)}")
            print(f"\nSorry, something went wrong: {str(e)}")


def main():
    try:
        chat_loop()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()