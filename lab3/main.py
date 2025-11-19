from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from form_dataset import combine_datasets, download_datasets
from get_accuracy import get_accuracy
from train_model import train_model


def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_class = torch.argmax(logits, dim=1).item()
    return pred_class + 1


save_text = False
run_base = True

if __name__ == '__main__':
    model_name = 'roberta-base'
    
    hint_message = 'Choose action (train or perform): '
    act = input(hint_message)
    while act.lower() not in {'train', 'perform'}:
        act = input(hint_message)
    
    num_classes = 10
    
    file_path = './lab3/input.txt'
    
    if act == 'train':
        datasets_names = [
            'vinayaks0n1/imdb-tv-show-reviews',
            'itsnobita/one-piece-live-action-imdb-reviews',
            'forgetabhi/dune-part-two-imdb-reviews',
            'shivvm/popular-movies-imdb-reviews-dataset',
            'fahadrehman07/movie-reviews-and-emotion-dataset'
        ]

        datasets_paths = download_datasets(datasets_names)
        dataset = combine_datasets(datasets_paths)
        print(dataset['label'].unique())
        
        if save_text:
            k = 100
            
            texts = dataset['text'][:k]

            with open(file_path, 'w', encoding='utf-8') as f:
                for t in texts:
                    f.write(t + '\n')

        train, test = train_test_split(dataset, test_size=0.1)
        split_dataset = DatasetDict({
            'train': Dataset.from_pandas(train),
            'test': Dataset.from_pandas(test)
        })
        
        base_acc, base_f1 = get_accuracy(model_name, split_dataset, num_classes)
        print(f'Base model:\nAccuracy - {base_acc}, F1 - {base_f1}')
        
        train_model(model_name, num_classes, split_dataset)
    else:
        model_base = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
        )
        tokenizer_base = AutoTokenizer.from_pretrained(model_name)
        
        path_to_model = './lab3/sentiment_classification/checkpoint-21027'
        
        model_trained = AutoModelForSequenceClassification.from_pretrained(
            path_to_model,
            num_labels=num_classes,
        )
        tokenizer_trained = AutoTokenizer.from_pretrained(path_to_model)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(*lines, sep='\n')
            
            if run_base:
                print('Base model:')
                for line in lines:
                    print(predict(line, model_base, tokenizer_base))
                
                print('---------------------------------------')
            
            # print('Trained model:')
            # for line in lines:
            #     print(predict(line, model_trained, tokenizer_trained))
            