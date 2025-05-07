import unittest
from src.pipeline import RAGPipeline

class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        documents = [
            "Q: What is the course duration? A: The course lasts 12 weeks.",
            "Q: Are there any prerequisites? A: Yes, basic knowledge of Python is required."
        ]
        self.pipeline = RAGPipeline(documents)
    
    def test_process_query(self):
        result = self.pipeline.process_query("How long is the course?")
        self.assertIn("query", result)
        self.assertIn("contexts", result)
        self.assertIn("answer", result)
        self.assertTrue(len(result["contexts"]) > 0)
        self.assertIn("12 weeks", result["answer"].lower())
    
    def test_query_preprocessing(self):
        result = self.pipeline.process_query("Q: What is the course duration?")
        self.assertIn("12 weeks", result["answer"].lower())
    
    def test_exact_answer_extraction(self):
        result = self.pipeline.process_query("What is the course duration?")
        self.assertEqual(result["answer"], "The course lasts 12 weeks.")

if __name__ == '__main__':
    unittest.main()