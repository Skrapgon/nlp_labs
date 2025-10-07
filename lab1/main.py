import os
from timeit import default_timer

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

if __name__ == '__main__':
    main()