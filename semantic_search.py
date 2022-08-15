from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import os
#from skimage.io import imread, imshow


#embedder = SentenceTransformer('all-MiniLM-L6-v2')
# List of models optimized for semantic textual similarity can be found at:
# https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0
#distiluse-base-multilingual-cased-v1
#model = SentenceTransformer('stsb-roberta-large')
#https://www.sbert.net/docs/pretrained_models.html

def get_similars(query):

    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    filenames = list(np.load("files/filenames.npy"))
    captions = list(np.load("files/captions.npy"))
    corpus_embeddings = torch.load('files/corpus_embeddings.pt')
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)[0]

    similar_files = []
    similar_captions = []
    for hit in hits:
        number = hit["corpus_id"]
        filename = filenames[number]
        caption = captions[number]
        similar_files.append(filename+".jpg")
        similar_captions.append(caption)

    return similar_files, similar_captions

