from train import train
import preprocess

train(lr=0.08, w=4, l2_reg=0.0004, epoch=2, batch_size=64,
    num_layers=2, data_type="WikiQA",
    word2vec=preprocess.Word2Vec())
