from flask import Flask, request, render_template
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from flask_ngrok import run_with_ngrok
from transformers import BartForConditionalGeneration, BartTokenizer
from langchain.prompts import PromptTemplate
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Determine the device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

DB_FAISS_PATH = 'diabetes_vectorstore/db_faiss'

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Initialize Flask app
app = Flask(__name__)
run_with_ngrok(app)

# Document Retriever Class
class DocumentRetriever:
    def __init__(self, faiss_path, model_name="intfloat/multilingual-e5-large", device="cpu"):
        self.faiss_path = faiss_path
        self.model_name = model_name
        self.device = device
        self.embeddings = None
        self.db = None

    def load_faiss_and_embeddings(self):
        """
        Load FAISS index and Hugging Face embeddings for retrieval.
        """
        try:
            logging.info("Initializing HuggingFace Embeddings...")
            self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs={'device': self.device})

            logging.info("Loading FAISS vector store...")
            self.db = FAISS.load_local(self.faiss_path, self.embeddings, allow_dangerous_deserialization=True)
            logging.info("FAISS vector store loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading FAISS or embeddings: {e}")
            raise

    def retrieve_documents(self, query, k=3):
        """
        Perform FAISS-based document retrieval for a given query and return 'k' documents.
        """
        if self.db is None:
            raise ValueError("FAISS database not loaded. Please load the FAISS vector store first.")

        try:
            logging.info(f"Retrieving {k} documents for the query: '{query}'")
            retrieved_docs = self.db.as_retriever(search_kwargs={'k': k}).get_relevant_documents(query)
            logging.info(f"Successfully retrieved {len(retrieved_docs)} documents.")

            if retrieved_docs:
                return [doc.page_content[:400] for doc in retrieved_docs]  # Return content of 'k' documents
            else:
                return []
        except Exception as e:
            logging.error(f"Error during document retrieval: {e}")
            raise


# BART Paraphrase Generator
# class BARTGenerator:
#     def __init__(self, bart_model_name='eugenesiow/bart-paraphrase', device="cpu"):
#         self.bart_model_name = bart_model_name
#         self.device = device

#         # Load BART model and tokenizer for paraphrasing
#         self.bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name).to(device)
#         self.bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)

#     def generate_response(self, context, question):
#         """
#         Generate a response using BART based on the retrieved document context and user question.
#         """
#         # Combine context and question as input for BART
#         input_text = f"{context} {question}"
        
#         # Tokenize input text
#         inputs = self.bart_tokenizer(input_text, return_tensors="pt", truncation=True).to(self.device)
        
#         # Generate paraphrase
#         outputs = self.bart_model.generate(inputs["input_ids"], max_length=300, num_beams=4, no_repeat_ngram_size=2, early_stopping=True)
        
#         # Decode the generated text
#         generated_response = self.bart_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return generated_response

class BARTGenerator:
    def __init__(self, bart_model_name='eugenesiow/bart-paraphrase', device="cpu"):
        self.bart_model_name = bart_model_name
        self.device = device

        # Load BART model and tokenizer for paraphrasing
        self.bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name).to(device)
        self.bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)

    def generate_response(self, context, question):
        """
        Generate a response using BART based on the retrieved document context and user question.
        Ensure only the answer portion is paraphrased.
        """
        # Extract the answer portion from the context by removing any questions
        answer_part = self.extract_answer(context)
        
        if not answer_part:
            return "No relevant answer found in the document."

        # Tokenize the answer portion
        inputs = self.bart_tokenizer(answer_part, return_tensors="pt", truncation=True).to(self.device)
        
        # Generate paraphrase
        outputs = self.bart_model.generate(inputs["input_ids"], max_length=500, num_beams=4, no_repeat_ngram_size=2, early_stopping=True)
        
        # Decode the generated text
        paraphrased_answer = self.bart_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return paraphrased_answer
    
    def extract_answer(self, text):
        """
        Extracts the answer portion of the retrieved document by ignoring questions.
        """
        # Heuristic to detect if the text is a question (if it contains question words or ends with a question mark)
        question_words = ["what", "how", "why", "when", "who", "where", "is", "are", "can", "should", "do", "does"]

        # Split the text into sentences
        sentences = text.split('.')
        answer_sentences = []
        
        for sentence in sentences:
            # Ignore sentences that are likely questions
            sentence = sentence.strip()
            if not (sentence.lower().startswith(tuple(question_words)) or sentence.endswith("?")):
                answer_sentences.append(sentence)
        
        # Reconstruct the answer part
        answer_part = '. '.join(answer_sentences).strip()
        return answer_part if answer_part else None


# Initialize the retriever and BART generator
retriever = DocumentRetriever(faiss_path=DB_FAISS_PATH, model_name="intfloat/multilingual-e5-large", device=device)
retriever.load_faiss_and_embeddings()

bart_generator = BARTGenerator(device=device)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query")
        if query:
            try:
                # Retrieve three relevant documents (contexts)
                contexts = retriever.retrieve_documents(query, k=2)

                if contexts:
                    # Generate a response using BART for each document
                    bart_responses = [bart_generator.generate_response(context=context, question=query) for context in contexts]
                else:
                    contexts = ["No relevant document found."]
                    bart_responses = ["I don't know the answer to this question based on the available information."]

                # Combine contexts and responses into one list of tuples
                combined_results = list(zip(contexts, bart_responses))

                # Pass the query, contexts, and bart_responses to the template
                return render_template("index.html", query=query, combined_results=combined_results, zip=zip)
            except Exception as e:
                return render_template("index.html", query=query, error=str(e))
    return render_template("index.html")




if __name__ == "__main__":
    # app.run(debug=True, port=5056)
    app.run()
