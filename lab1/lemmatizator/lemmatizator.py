import pickle
import os, re

from sys import getsizeof
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage

class DictLemmatizator:
    def __init__(self):
        load_dotenv()
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
            'INFN': 9,
            'NOUN': 10
        }
        
        self.credentials = os.getenv('API_TOKEN')
        
        llm = HuggingFaceEndpoint(
            repo_id='openai/gpt-oss-20b',
            provider='nscale',
            # repo_id='ruslandev/llama-3-8b-gpt-4o-ru1.0',
            # provider='featherless-ai',
            max_new_tokens=128,
            temperature=0.02,
            huggingfacehub_api_token=self.credentials,
        )
        self.llm = ChatHuggingFace(llm=llm)
    
         
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
    
    
    def llm_find(self, lemmas, sentence):
        system_message = f'''
        алгоритм нашел для каждого слова из текста словарную форму и часть речи:
        {lemmas}
        в данном контексте:
        {sentence}
        Для каждого слова выдать ответ на отдельной строке.
        Если верно, то в ответе поставь только знак +.
        ЕСЛИ неверно, то в ответе напиши через пробел правильную словарную форму и часть
        речи из списка, представленного ниже:
        'NOUN', 'ADJF', 'INFN', 'NUMR', 'ADVB',
        'NPRO', 'PREP', 'CONJ', 'PRCL', 'INTJ'.
        ТВОИ ОТВЕТЫ МОГУТ БЫТЬ ТОЛЬКО ТАКИМИ:
        +
        верная_форма ВЕРНАЯ_ЧАСТЬ_РЕЧИ
        НИКАКИХ ДРУГИХ СИМВОЛОВ И ФОРМАТОВ ОТВЕТА НЕ ДОЛЖНО БЫТЬ В ОТВЕТЕ.
        НИКАКИХ НОМЕРОВ СТРОК.
        Помни, что словарная форма для глагола - инфинитив, для существительного -
        именительный падеж, ед. число, для прилагательного - оно в им. падеже, единственном
        числе, в мужском роде.
        Например, для "гулял Андрей" ответ:
        гулять INFN
        андрей NOUN
        '''

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=f'Проверь слова {lemmas} в контексте: {sentence}')
        ]

        try:
            llm_response = self.llm.invoke(messages).content.strip()
            
            print(llm_response, '\n----------------------')

            responses = llm_response.split('\n')
            res = []
            for i in range(len(lemmas)):
                if responses[i] == '+':
                    res.append(lemmas[i])
                else:
                    parts = responses[i].split()
                    if len(parts) >= 2:
                        new_lemma = parts[0].lower()
                        new_pos = parts[1]
                        word = lemmas[i].split('{')[0]
                        res.append(f'{word}{{{new_lemma}={new_pos}}}')
                    else:
                        res.append(lemmas[i])
            return res
        except Exception as e:
            print(f'Ошибка: {e}')
            return lemmas
    
    
    def lemmatize_text(self, text: str, first: bool = True, llm: bool = True) -> str:
        sentences = text.split('.')
        result = []
        for sentence in sentences:
            
            words = re.split('[ ,\!\?«»()—";:\n]', sentence)
            
            sentence_lemmas = []
            
            for word in words:
                if word == '':
                    continue
                
                word_lemm = self.find_word(word, first | llm)
                if isinstance(word_lemm, set):
                    sentence_lemmas.append(f'{word}{{{', '.join([f'{wl[0]}={wl[1]}' for wl in word_lemm])}}}')
                else:
                    sentence_lemmas.append(f'{word}{{{word_lemm[0]}={word_lemm[1]}}}')
            if llm:
                sentence_lemmas = self.llm_find(sentence_lemmas, sentence)
            result.extend(sentence_lemmas)
        return ' '.join(result)