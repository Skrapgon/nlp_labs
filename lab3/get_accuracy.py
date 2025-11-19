from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_accuracy(model_name, dataset, num_classes, batch_size=32):
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_ds = dataset['test']

    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch])
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        return inputs, labels

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return accuracy, f1