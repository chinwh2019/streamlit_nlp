import torch
from sentence_transformers import SentenceTransformer, util

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
# model = SentenceTransformer('all-mpnet-base-v2')
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')


#Our sentences we like to encode
# sentences = ['This framework generates embeddings for each input sentence',
#     'Sentences are passed as a list of string.',
#     'The quick brown fox jumps over the lazy dog.']
#
# #Sentences are encoded by calling model.encode()
# sentence_embeddings = model.encode(sentences)
#
# #Print the embeddings
# for sentence, embedding in zip(sentences, sentence_embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", embedding)
#     print("")
#
# #Sentences are encoded by calling model.encode()
# emb1 = model.encode("サンディにお菓子をプレゼント")
# emb2 = model.encode("サンディにお菓子を配った")
# cos_sim = util.cos_sim(emb1, emb2)
# print("Cosine-Similarity:", cos_sim)

# Semantic Search
query_embedding = model.encode('How big is London')
passage_embedding = model.encode(['London has 9,787,426 inhabitants at the 2011 census',
                                  'London is known for its finacial district'])

print("Similarity:", util.dot_score(query_embedding, passage_embedding))

