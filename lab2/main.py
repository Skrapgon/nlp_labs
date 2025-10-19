import json
import os

from datasets import load_dataset

from summarizer import Summarizer
from get_acc import Evaluator

ruT5 = 'cointegrated/ruT5-base'
UrukHan_t5 = 'UrukHan/t5-russian-summarization'
ru_sum = 'sarahai/ru-sum'

dataset_name = 'IlyaGusev/gazeta'
revision = 'v2.0'

max_chars = 300
max_length = 90

num = 10

max_texts = 200

base_path = os.path.dirname(__file__)
input_file = 'input.json'
output_file = 'output.json'

def find_best_model(save_texts: bool = False):
    evaluator = Evaluator()
    
    dataset = load_dataset(dataset_name, revision=revision, trust_remote_code=True)
    filtered_dataset = dataset.filter(lambda row: len(row['summary']) <= 300)
    test_length = len(filtered_dataset['test']) if max_texts <= 0 else min(max_texts, len(filtered_dataset['test']))
    
    texts = [filtered_dataset['test'][i]['text'] for i in range(test_length)]
    summaries = [filtered_dataset['test'][i]['summary'] for i in range(test_length)]
    if save_texts:
        save_json(texts[:num])
        save_json(summaries[:num], 'gold.json')
    
    summarizer1 = Summarizer(UrukHan_t5)
    summarizer2 = Summarizer(ru_sum)
    
    print('Models are loaded')
    
    preds1 = [summarizer1.summarize_text(text) for text in texts]
    scores1 = evaluator.evaluate(summaries, preds1)
    print('First model is evaluated')
    
    preds2 = [summarizer2.summarize_text(text) for text in texts]
    scores2 = evaluator.evaluate(summaries, preds2)
    print('Second model is evaluated')
    
    print(scores1[0])
    print('AVG score:')
    print(scores1[1])
    print('-------------------')
    print(scores2[0])
    print('AVG score:')
    print(scores2[1])
    
def save_json(texts, file=input_file):
    with open(os.path.join(base_path, file), 'w', encoding='utf-8') as save_file:
        json.dump(texts, save_file)

def main(best_model):
    with open(os.path.join(base_path, input_file), 'r') as inp_file:
        texts = json.load(inp_file)
        
        summarizer = Summarizer(best_model)
        results = [summarizer.summarize_text(text) for text in texts]
        
        print(texts)
        print(results)
        
        with open(os.path.join(base_path, output_file), 'w') as save_file:
            json.dump(results, save_file)
        
        with open(os.path.join(base_path, 'gold.json'), 'r') as gold:
            summ = json.load(gold)
            print(summ)

if __name__ == '__main__':
    find_best_model(save_texts=True)
    main(ru_sum)