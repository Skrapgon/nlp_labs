from rouge import Rouge

class Evaluator:
    def __init__(self):
        self.rouge = Rouge()
    
    def evaluate(self, references, predictions):
        scores = [ev['rouge-2'] for ev in self.rouge.get_scores(references, predictions)]
        scores_avg = self.rouge.get_scores(references, predictions, avg=True)['rouge-2']
        return scores, scores_avg