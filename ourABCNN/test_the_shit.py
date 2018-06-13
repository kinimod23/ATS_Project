from train import train
import preprocess

train(lr=0.08, w=4, l2_reg=0.0004, epoch=201, batch_size=128,
    num_layers=2, data_type="Complex2Simple", method='labeled',
    word2vec=preprocess.Word2Vec(), pickleData=False)
