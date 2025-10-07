import os
import pickle
import xml.etree.ElementTree as ET

words: dict[str, int | list[int]] = {} # все слова
lemmas: dict[int, list[str, int]] = {} # леммы
speech_parts: list[str] = [] # части речи

links_dict: dict[int, int | set[int]] = {} # связь между некоторым формами слов (для поиска начальной формы глаголов и прочего)

tree = None
root = None

def load_xml(file_name: str):
    global tree
    global root
    tree = ET.parse(file_name)
    root = tree.getroot()


def save_speech_parts():
    grammemes = root.find('grammemes')

    for g in grammemes.findall('grammeme'):
        parent = g.get('parent')
        if parent != 'POST':
            continue
        name = g.find('name').text
        speech_parts.append(name)

    with open('speech_parts.pkl', 'wb') as speech_parts_file:
        pickle.dump(speech_parts, speech_parts_file)


def get_lemmas_id(id: int) -> set[int]:
    stack = [id]
    result = set()
    while stack:
        cur = stack.pop()
        if cur in links_dict:
            ids = links_dict[cur]
            if isinstance(ids, set):
                stack.extend(ids)
            else:
                stack.append(ids)
        else:
            result.add(cur)
            
    return result


def save_words_lemmas():    
    links = root.find('links')
    
    for l in links.findall('link'):
        to = int(l.get('to'))
        fr = int(l.get('from'))
        if to not in links_dict:
            links_dict[to] = fr
        else:
            tmp = links_dict[to]
            if isinstance(tmp, set):
                tmp.add(fr)
            else:
                links_dict[to] = set([tmp, fr])
    
    lemmata = root.find('lemmata')
    
    for lem in lemmata.findall('lemma'):
        id = int(lem.get('id'))
        
        l = lem.find('l')
        lemma = l.get('t')
        lemma = lemma.replace('ё', 'е')
        ps = speech_parts.index(l.find('g').get('v'))
        
        lemm_id = get_lemmas_id(id)
        
        if id in lemm_id:
            lemmas[id] = [lemma, ps]
        
        tmp = lemm_id.pop()
        lemm_id.add(tmp)
        
        for g in lem.findall('f'):
            word = g.get('t')
            word = word.replace('ё', 'е')
            
            if word not in words:
                words[word] = tmp if len(lemm_id) == 1 else lemm_id
            else:
                if isinstance(words[word], set):
                    words[word].update(lemm_id)
                else:
                    words[word] = set([words[word], *lemm_id])
    
    with open('lemmas.pkl', 'wb') as lemmas_file:
        pickle.dump(lemmas, lemmas_file)
        
    with open('words.pkl', 'wb') as words_file:
        pickle.dump({k: list(v) if isinstance(v, set) and len(v) > 1 else list(v)[0] if isinstance(v, set) and len(v) == 1 else v for k, v in words.items()}, words_file)
    

base_path = os.path.dirname(__file__)

load_xml(os.path.join(base_path, 'dict.opcorpora.xml'))
save_speech_parts()
save_words_lemmas()