from src.retrieval import Retriever
from src.generation import Generator
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGPipeline:
    def __init__(self, documents):
        self.retriever = Retriever(documents)
        self.generator = Generator()
    
    def process_query(self, query):
        # Retrieve relevant documents
        retrieved = self.retriever.retrieve(query, k=5, threshold=0.1)
        # Extract full contexts for display, but use highest-scoring for generation
        full_contexts = [doc for doc, _ in retrieved]
        scores = [score for _, score in retrieved]
        
        # Use the highest-scoring context for generation
        contexts = [full_contexts[0]] if full_contexts else []
        
        # Log retrieval
        logging.info(f"Query: {query}")
        logging.info(f"Retrieved contexts: {full_contexts}")
        logging.info(f"Retrieval scores: {scores}")
        
        # Generate answer
        answer = self.generator.generate(query, contexts)
        logging.info(f"Generated answer: {answer}")
        
        return {
            "query": query,
            "contexts": retrieved,  # Store full contexts for display
            "scores": scores,
            "answer": answer
        }