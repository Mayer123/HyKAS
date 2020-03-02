import json 
import csv
import tqdm
from collections import Counter
import pickle 

def read_commonsense(file):
    en_concepts = {}
    rel_types = {}
    long_en_concepts = {}
    word_pos = {}
    concept_len = Counter()
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, line in enumerate(reader):
            if line[2].split('/')[2] == 'en':
                if line[3].split('/')[2] == 'en':
                    rela = '/'.join(line[1].split('/')[2:]) 
                    if rela not in rel_types:
                        rel_types[rela] = 1
                    start = line[2].split('/')[3]
                    end = line[3].split('/')[3]
                    concept_len[len(start.split('_'))] += 1
                    concept_len[len(end.split('_'))] += 1
                    if len(line[2].split('/')) == 4:
                        start_sense = ''
                    else:
                        start_sense = line[2].split('/')[4]
                    if start_sense not in word_pos:
                        word_pos[start_sense] = 1
                    if len(line[3].split('/')) == 4:
                        end_sense = ''
                    else:
                        end_sense = line[3].split('/')[4]
                    if end_sense not in word_pos:
                        word_pos[end_sense] = 1
                    meta = json.loads(line[-1])
                    concept = (start_sense, rela, end, end_sense, meta['weight'])
                    if len(start.split('_')) > 1:
                        for w in start.split('_'):
                            if w not in long_en_concepts:
                                long_en_concepts[w] = {start:[concept]}
                            elif start not in long_en_concepts[w]:
                                long_en_concepts[w][start] = [concept]
                            else:
                                long_en_concepts[w][start].append(concept)
                    else:
                        if start not in en_concepts:
                            en_concepts[start] = [concept]
                        else:
                            en_concepts[start].append(concept)
            if i % 5000000 == 0:
                print (i)
    print (len(en_concepts))
    print (len(long_en_concepts))
    print (word_pos)
    #print (rel_types)
    #print (concept_len)
    return en_concepts, long_en_concepts

def save_dict(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    en_concepts, long_en_concepts = read_commonsense('conceptnet-assertions-5.6.0.csv')
    save_dict('en_concepts.pickle', en_concepts)
    save_dict('long_en_concepts.pickle', long_en_concepts)