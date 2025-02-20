from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
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


class ChatBot:
    def __init__(self, persist_directory: str = "chroma_db", collection_name: str = "documents"):
        """Initialize chatbot with ChromaDB connection and memory"""
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize components
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            streaming=True
        )

        self._setup_database()
        self._setup_memory()
        self._setup_chain()

    def _setup_database(self):
        """Set up ChromaDB connection"""
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            logger.info(f"Connected to ChromaDB at {self.persist_directory}")
        except Exception as e:
            logger.error(f"ChromaDB connection failed: {str(e)}")
            raise

    def _setup_memory(self):
        """Set up conversation memory"""
        try:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key='answer'
            )
            logger.info("Chat memory initialized")
        except Exception as e:
            logger.error(f"Memory setup failed: {str(e)}")
            raise

    def _setup_chain(self):
        """Set up the conversational chain"""
        try:
            # Custom prompt template
            CUSTOM_PROMPT = PromptTemplate.from_template(
                """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. 
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                Maintain a conversational tone and refer back to previous parts of the conversation when relevant.

                {context}

                Question: {question}
                Helpful Answer:"""
            )

            # Create the chain
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                ),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
                return_source_documents=True,
                verbose=True
            )
            logger.info("Conversational chain setup complete")
        except Exception as e:
            logger.error(f"Chain setup failed: {str(e)}")
            raise

    def ask(self, question: str) -> str:
        """Process a question and return an answer"""
        try:
            logger.info(f"Received question: {question}")
            response = self.chain.invoke(question)
            logger.info("Generated response successfully")
            return response["answer"]
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

    def reset_memory(self):
        """Reset the chat memory"""
        try:
            self._setup_memory()
            self._setup_chain()
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