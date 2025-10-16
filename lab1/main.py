import os
from timeit import default_timer
import json

from get_acc import evaluate_morph_analyzer
from lemmatizator.lemmatizator import DictLemmatizator

def main():
    lemmatizator = DictLemmatizator()
    base_path = os.path.dirname(__file__)
    with open(os.path.join(base_path, 'input.txt'), 'r', encoding='utf-8') as input:    
        with open(os.path.join(base_path, 'output.txt'), 'w', encoding='utf-8') as output:
            for line in input.readlines():
                output.write(lemmatizator.lemmatize_text(line))
                output.write('\n')

def calculate_avg_initialization_time(iter: int = 10):
    start = default_timer()
    for _ in range(iter):
        DictLemmatizator()
    end = default_timer()
    
    print(f'AVG initialization time: {(end-start)/iter}')

def calculate_avg_lemmatization_time(file: str = 'test_lemmatization_time.txt', iter: int = 10):
    lemmatizator = DictLemmatizator()
    base_path = os.path.dirname(__file__)
    with open(os.path.join(base_path, file), 'r', encoding='utf-8') as input:    
        line = input.readline()
        start = default_timer()
        for _ in range(iter):
            lemmatizator.lemmatize_text(line)
        end = default_timer()
        print(f'AVG lemmatization time: {(end-start)/iter}')

def evaluate_lemmatizator_acc(path: str = 'evaluate.json'):
    base_path = os.path.dirname(__file__)
    with open(os.path.join(base_path, path), 'br') as json_file:
        lemmatizator = DictLemmatizator()
        total = 0
        correct_lemma = 0
        correct_pos = 0
        correct_both = 0
        data = json.load(json_file)
        for d in data:
            text = d['text']
            markup = d['markup']
            markup_text = lemmatizator.lemmatize_text(text, True, False)
            cur_total, cur_correct_lemma, cur_correct_pos, cur_correct_both = evaluate_morph_analyzer(text, markup_text, markup)
            total += cur_total
            correct_lemma += cur_correct_lemma
            correct_pos += cur_correct_pos
            correct_both += cur_correct_both
        print(f'Всего токенов: {total}')
        print(f'Итоговая точность по лемме: {correct_lemma/total*100:.4f}%')
        print(f'Итоговая точность по POS: {correct_pos/total*100:.4f}%')
        print(f'Итоговое совпадение лемма + POS: {correct_both/total*100:.4f}%')

if __name__ == '__main__':
    # evaluate_lemmatizator_acc()
    main()
