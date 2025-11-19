import os
from glob import glob

import kagglehub

import pandas as pd

def download_datasets(datasets: list[str]):
    paths = []
    for dataset in datasets:
        paths.append(kagglehub.dataset_download(dataset))
    
    return paths


def process_dataset(path, 
                    text_cols=['review', 'reviews', 'review content'], 
                    label_cols=['rating', 'ratings', 'review_rating', 'user rating', 'rating (out of 10)'],
                    target_text='text',
                    target_label='label'):

    if path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        return

    text_col = next((c for c in df.columns if c.lower() in text_cols), None)
    if not text_col:
        return

    label_col = next((c for c in df.columns if c.lower() in label_cols), None)
    if not label_col:
        return

    df = df[[text_col, label_col]].copy()

    df = df.rename(columns={
        text_col: target_text,
        label_col: target_label
    })

    df[target_text] = df[target_text].astype(str).str.strip().replace('\n', ' ')
    try:
        df[target_label] = df[target_label].astype(int)
    except:
        pass

    df = df.dropna(subset=[target_text, target_label])
    df = df[df[target_text].str.len() > 0]
    
    df[target_label] -= 1

    return df


def combine_datasets(paths):
    all_dfs = []

    for path in paths:
        all_files = glob(os.path.join(path, '*'))
        for file in all_files:
            print(f'Обработка: {os.path.basename(file)}')
            df = process_dataset(file)
            if df is None:
                continue

        all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f'Получен общий датасет: {len(combined)} строк')
        return combined