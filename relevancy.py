import numpy as np
import spacy
from nltk import ngrams
from sentence_transformers import SentenceTransformer, util


# Initializiation
model = SentenceTransformer('all-mpnet-base-v2')
nlp = spacy.load('en_core_web_sm')


def top_match(reqs, vitae, avg_over_k=30):
    grams_list = []
    for vit in vitae:
        grams = list(ngrams(vit.split(), 4))
        for x in range(0,len(grams), 3):
            gram = grams[x]
            grams_list.append(' '.join(gram))
        if len(grams) % 3 !=0:
            grams_list.append(' '.join(grams[-1]))
    vitae = grams_list

    reqs_embd = model.encode(reqs, convert_to_tensor=True)
    vitae_embd = model.encode(vitae, convert_to_tensor=True)
    cosine_scores = util.cos_sim(reqs_embd, vitae_embd)
    cosine_scores = cosine_scores.numpy()
    top_k = np.sort(np.ravel(cosine_scores))[-avg_over_k:]
    
    results = {x: [] for x in vitae}
    for top in reversed(top_k):
        itemindex = np.where(cosine_scores == top)
        if len(itemindex[0]) == 1:
            itemindex = [itemindex]
        for it_dex in itemindex:
            index_reqs, index_vitae = int(it_dex[0]), int(it_dex[1])
            results[vitae[index_vitae]].append(float(top))
    avg_dict = {}
    for key, value in results.items():
        if value:
            avg_dict[key] = np.mean(value)
    avg_dict = sorted(avg_dict.items(), key=lambda x: x[1], reverse=True)
    return avg_dict
