from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import re

class Generator:
    def __init__(self, model_name="facebook/bart-base"):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def generate(self, query, contexts):
        # Use the first context as the primary answer source
        primary_context = contexts[0] if contexts else ""
        input_text = f"Extract the answer from the following context. Return only the text after 'A:'. If no answer is found, return 'No clear answer found.'\n\nContext: {primary_context}"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, max_length=100, num_beams=4, early_stopping=True)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the answer text after 'A:'
        answer_match = re.search(r'A:\s*(.*?)(?:\n|$)', answer, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            answer = "No clear answer found."
        
        # Clean up any remaining artifacts
        answer = re.sub(r'^(Extract|Context).*?\n?', '', answer, flags=re.IGNORECASE)
        return answer if answer else "No clear answer found."