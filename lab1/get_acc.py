def evaluate_morph_analyzer(text, marked_text, gold_data):
    total = 0
    correct_lemma = 0
    correct_pos = 0
    correct_both = 0
    
    words = marked_text.split()

    for i in range(len(gold_data)):
        parts = words[i].split('{')
        
        lemma_pos = parts[1]
        lemma_pos_list = lemma_pos.split('=')
        lemma = lemma_pos_list[0]
        pos = lemma_pos_list[1][:-1]

        total += 1
        if lemma == gold_data[i]['lemma']:
            correct_lemma += 1
        if pos == gold_data[i]['pos']:
            correct_pos += 1
        if lemma == gold_data[i]['lemma'] and pos == gold_data[i]['pos']:
            correct_both += 1

    lemma_acc = correct_lemma / total * 100
    pos_acc = correct_pos / total * 100
    both_acc = correct_both / total * 100

    print(f'Исходный текст: {text}')
    print(f'Всего токенов: {total}')
    print(f'Точность по лемме: {lemma_acc:.4f}%')
    print(f'Точность по POS: {pos_acc:.4f}%')
    print(f'Совпадение лемма + POS: {both_acc:.4f}%')
    print('--------------------------------')
    return total, correct_lemma, correct_pos, correct_both