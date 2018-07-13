from gensim.models.fasttext import FastText
import time
from gensim.models import KeyedVectors

start = time.time()

test_save="wiki.dump"
embeddings="wiki.en.bin"


model = FastText.load_fasttext_format(embeddings, encoding='utf8')
#model = FastText.load(embeddings)
model.save(test_save)

mid = time.time()
print("Loading the model took: {} seconds.".format(mid-start))

print("Test:\n")
print(model.most_similar("dog"))

