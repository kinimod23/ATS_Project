import tensorflow as tf
import numpy as np
import sys

from preprocess_dump2 import Word2Vec, ComplexSimple, FastText
from ABCNN_splitted import ABCNN_conv, ABCNN_deconv
from utils import build_path
import pickle
import os
import gensim


def test(w, l2_reg, epoch, max_len, model_type, data, word2vec, num_layers, num_classes=2):

############################################################################
#########################   DATA LOADING   #################################
############################################################################
    if model_type == 'convolution': method = 'labeled'
    else: method = 'unlabeled'
    dumped = 'preprocessed_test_'+method+'_'+data+'_'+word2vec+'.pkl'
    if word2vec == 'FastText': w2v = FastText()
    else: w2v = Word2Vec()

    if not os.path.exists(dumped):
        print("Dumped data not found! Data will be preprocessed")
        test_data = ComplexSimple(word2vec=w2v)
        test_data.open_file(mode="test", method=method, data=data, word2vec=word2vec)
    else:
        print("found pickled state, loading..")
        test_data = ComplexSimple(word2vec=w2v)
        with open(dumped, 'rb') as f:
            dump_dict = pickle.load(f)
            for k, v in dump_dict.items():
                setattr(test_data, k, v)
        print("done!")

############################################################################
#########################      MODEL      ##################################
############################################################################
    tfconfig = tf.ConfigProto(allow_soft_placement = True)
    with tf.device("/gpu:0"):
        encoder = ABCNN_conv(s=test_data.max_len, w=w, l2_reg=l2_reg,
                  num_layers=num_layers)

    model_path = build_path("./models/", data, 'BCNN', num_layers, model_type, word2vec)
    print("=" * 50)
    print("test data size:", test_data.data_size)

############################################################################
#########################     TRAINING     #################################
############################################################################
    Accuracys = []
    with tf.Session(config=tfconfig) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path + "-" + str(1000))
        print(model_path + "-" + str(1000), "restored.")

        Sentences = []
        for e in range(1, epoch + 1):
            print("[Epoch " + str(e) + "]")
            test_data.reset_index()
            i, MeanCost = 0, 0
            s1s, s2s, labels = test_data.next_batch(batch_size=test_data.data_size)
            for i in range(test_data.data_size):
                pred, c, a = sess.run([encoder.prediction, encoder.cost, encoder.acc],
                                           feed_dict={encoder.x1: np.expand_dims(s1s[i], axis=0),
                                                      encoder.x2: np.expand_dims(s2s[i], axis=0),
                                                      encoder.y1: np.expand_dims(labels[i], axis=0)})
                MeanCost += c
                Accuracys.append(a)
                Sentences.append(pred)
                if i % 200 == 0:
                    print('[batch {}]  cost: {}  accuracy: {}'.format(i, c, a))
            print('Mean Cost: {}   Mean Accuracy: {}'.format(MeanCost/i, np.mean(Accuracys)))

    print("=" * 50)
    print("max accuracy: {}  mean accuracy: {}".format(max(Accuracys), np.mean(Accuracys)))
    print("=" * 50)
    print('Number of Sentences: {}'.format(len(Sentences)))

    fasttext = gensim.models.KeyedVectors.load("wiki.dump")
    print('FastText loaded')
    with open('output.txt', 'w') as f:
        for sen in Sentences[:2]:
            string = ''
            for word in range(50):
                string += fasttext.wv.most_similar(positive=sen[0][:,word,:].T, topn=1)[0][0] + ' '
            string += '\n'
            f.write(string)
    print('Output created!')

if __name__ == "__main__":

    # Paramters
    # --ws: window_size
    # --l2_reg: l2_reg modifier
    # --epoch: epoch
    # --max_len: max sentence length
    # --model_type: model type
    # --num_layers: number of convolution layers
    # --data_type: MSRP or WikiQA data
    # --classifier: Final layout classifier(model, LR, SVM)

    # default parameters
    params = {
        "ws": 4,
        "l2_reg": 0.0004,
        "epoch": 1,
        "model_type": "End2End",
        "max_len": 40,
        "num_layers": 4,
        "data": 'Wiki',
        "word2vec": 'FastText'
    }

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            k = arg.split("=")[0][2:]
            v = arg.split("=")[1]
            params[k] = v

    test(w=int(params["ws"]), l2_reg=float(params["l2_reg"]), epoch=int(params["epoch"]),
         max_len=int(params["max_len"]), model_type=params["model_type"],
         num_layers=int(params["num_layers"]), data=params["data"], word2vec=params["word2vec"])

