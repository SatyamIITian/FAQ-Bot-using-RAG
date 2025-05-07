# FAQ-Bot-using-RAG
A lightweight Retrieval-Augmented Generation (RAG) system for answering course-related FAQs with exact answers from the FAQ dataset.
Setup in GitHub Codespace

Clone the repository.
Open in Codespace.
Run pip install -r requirements.txt.
Ensure data/faqs.txt exists with FAQ content.
Run python main.py to execute the pipeline.

Dependencies

Python 3.10
scikit-learn
transformers
torch
numpy
pandas

Project Structure

data/: Contains faqs.txt with course FAQs.
src/: Source code for retrieval, generation, and pipeline.
tests/: Unit tests for the pipeline.
main.py: Main script to run the interactive FAQ bot, pipeline, and evaluate results.

Running the Bot
python main.py


The bot starts in interactive mode, prompting for user queries.
Type a question (e.g., "What is the course duration?") to get the exact answer from faqs.txt (e.g., "The course lasts 12 weeks, with weekly lectures and assignments.").
Type exit to stop interactive mode and run the evaluation for predefined queries.
Evaluation results are saved in evaluation_results.csv.

Features

Retrieval: TF-IDF with full FAQ indexing and query preprocessing for precise context matching.
Generation: Uses facebook/bart-base to extract exact answers from the highest-scoring FAQ.
Evaluation: Robust metrics for relevance, completeness, and faithfulness.
Interactive Mode: User-friendly interface with source highlighting and low-confidence warnings.

Running Tests
python -m unittest tests/test_pipeline.py

Evaluation
The system is tested with 10 predefined queries, and results are evaluated for:

Relevance: Are retrieved contexts relevant to the query?
Completeness: Is the answer sufficiently detailed?
Faithfulness: Does the answer align with retrieved contexts?

Results are saved in evaluation_results.csv.
