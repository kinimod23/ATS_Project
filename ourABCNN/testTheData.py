from preprocess import ComplexSimple, Word2Vec

train_data = ComplexSimple(word2vec=Word2Vec())
train_data.open_file(mode="train", method='labeled')

batch_x1, batch_x2, batch_y, batch_features = train_data.next_batch(batch_size=1)
