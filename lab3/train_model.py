import os
from datasets import Value
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


def get_last_checkpoint(path):
    checkpoints = [d for d in os.listdir(path) if d.startswith('checkpoint-')]
    if not checkpoints:
        return None
    
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))
    return os.path.join(path, checkpoints[-1])


def train_model(mode_name: str, num_classes: int, dataset, epochs: int = 3):
    checkpoint_dir = './lab3/sentiment_classification'
    last_ckpt = get_last_checkpoint(checkpoint_dir)

    if last_ckpt:
        model_path = last_ckpt
    else:
        model_path = mode_name

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    
    def preprocess(examples):
        enc = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=256
        )
        enc['labels'] = examples['label']
        return enc


    encoded = dataset.map(preprocess, batched=True)
    encoded = encoded.cast_column('labels', Value('int64'))
    encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_classes
    )


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='weighted'),
        }


    args = TrainingArguments(
        output_dir=checkpoint_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        load_best_model_at_end=False,
        overwrite_output_dir=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded['train'],
        eval_dataset=encoded['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()