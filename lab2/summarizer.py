import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Summarizer:
    def __init__(self, model_name: str, max_chars: int = 300):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.max_chars = max_chars
        
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 30, num_beams: int = 4) -> str:
        inputs = self.tokenizer(
            text,
            return_tensors='pt'
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            repetition_penalty=3.0,
            length_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        result = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return result[:self.max_chars]