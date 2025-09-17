import json
from itertools import product
import os, re

from sys import getsizeof

class DictLemmatizator:
    def __init__(self):
        base_path = os.path.dirname(__file__)
        with open(os.path.join(base_path, 'speech_parts.json'), 'r') as file:
            self.speech_parts = json.load(file)
        
        with open(os.path.join(base_path, 'lemmas.json'), 'r') as file:
            self.lemmas = json.load(file)

        with open(os.path.join(base_path, 'words.json'), 'r') as file:
            self.words = json.load(file)
            
    @property
    def speech_parts_size(self) -> int:
        return getsizeof(self.speech_parts)
    
    
    @property
    def lemmas_size(self) -> int:
        return getsizeof(self.lemmas)
    
    
    @property
    def words_size(self) -> int:
        return getsizeof(self.words)
    

    def generate_e_variants(self, s: str):
        options = [
            ['е', 'ё'] if ch == 'е' or ch == 'ё' else
            [ch]
            for ch in s
        ]
        
        return {''.join(p) for p in product(*options)}


    def find_word(self, word: str) -> tuple[str, str] | set[tuple[str, str]]:
        low_word = word.lower()
        ee_forms = self.generate_e_variants(low_word)
        for ee_word in ee_forms:
            if ee_word in self.words:
                if isinstance(self.words[ee_word], list):
                    return {(self.lemmas[str(w)][0], self.speech_parts[self.lemmas[str(w)][1]]) for w in self.words[ee_word]}
                else:
                    return (self.lemmas[str(self.words[ee_word])][0], self.speech_parts[self.lemmas[str(self.words[ee_word])][1]])
        return ('ERR', 'Word not found')
    
    
    def lemmatize_text(self, text: str) -> str:
        words = re.split('[ ,.\!\?«»()—"";:\n]', text)
        result = []
        
        for word in words:
            if word == '':
                continue
            
            result = result
            
            word_lemm = self.find_word(word)
            if isinstance(word_lemm, set):
                result.append(f'{word}{{{', '.join([f'{wl[0]}={wl[1]}' for wl in word_lemm])}}}')
            else:
                result.append(f'{word}{{{word_lemm[0]}={word_lemm[1]}}}')
        
        return ' '.join(result)