import pickle
import os, re

from sys import getsizeof

class DictLemmatizator:
    def __init__(self):
        base_path = os.path.dirname(__file__)
        with open(os.path.join(base_path, 'speech_parts.pkl'), 'rb') as file:
            self.speech_parts = pickle.load(file)
        
        with open(os.path.join(base_path, 'lemmas.pkl'), 'rb') as file:
            self.lemmas = pickle.load(file)

        with open(os.path.join(base_path, 'words.pkl'), 'rb') as file:
            self.words = pickle.load(file)
        
        self.pos_priority = {
            'PREP': 1,
            'CONJ': 2,
            'PRCL': 3,
            'INTJ': 4,
            'ADVB': 5,
            'NPRO': 6,
            'NUMR': 7,
            'ADJF': 8,
            'INF': 9,
            'NOUN': 10
        }
    
         
    @property
    def speech_parts_size(self) -> int:
        return getsizeof(self.speech_parts)
    
    
    @property
    def lemmas_size(self) -> int:
        return getsizeof(self.lemmas)
    
    
    @property
    def words_size(self) -> int:
        return getsizeof(self.words)
    
    
    def choose_best_sp(self, variants) -> str:
        return min(variants, key=lambda x: self.pos_priority.get(x, 99))


    def find_word(self, word: str, first: bool) -> tuple[str, str] | set[tuple[str, str]]:
        low_word = word.lower().replace('ё', 'е')
        if low_word in self.words:
            if isinstance(self.words[low_word], list):
                res = {(self.lemmas[w][0], self.speech_parts[self.lemmas[w][1]]) for w in self.words[low_word]}
                if first:
                    speech_part = self.choose_best_sp({v[1] for v in res})
                    for v in res:
                        if v[1] == speech_part:
                            return v
                return res
            else:
                return (self.lemmas[self.words[low_word]][0], self.speech_parts[self.lemmas[self.words[low_word]][1]])
        return ('UNK', low_word)
    
    
    def lemmatize_text(self, text: str, first: bool = True) -> str:
        words = re.split('[ ,.\!\?«»()—"";:\n]', text)
        result = []
        
        for word in words:
            if word == '':
                continue
            
            result = result
            
            word_lemm = self.find_word(word, first)
            if isinstance(word_lemm, set):
                result.append(f'{word}{{{', '.join([f'{wl[0]}={wl[1]}' for wl in word_lemm])}}}')
            else:
                result.append(f'{word}{{{word_lemm[0]}={word_lemm[1]}}}')
        
        return ' '.join(result)