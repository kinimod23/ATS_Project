import tensorflow as tf
import numpy as np
import sys

from preprocess_dump import MSRP, WikiQA
from preprocess_dump2 import Word2Vec, ComplexSimple, FastText
from ABCNN_reduced import ABCNN
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
        model = ABCNN(s=test_data.max_len, w=w, l2_reg=l2_reg, model_type=model_type,
                  num_features=test_data.num_features, num_classes=num_classes, num_layers=num_layers)

    model_path = build_path("./models/", 'BCNN', num_layers, model_type, word2vec)
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
            s1s, s2s, labels, features = test_data.next_batch(batch_size=test_data.data_size, model_type=model_type)
            for i in range(test_data.data_size):
                pred, c, a = sess.run([model.prediction, model.cost, model.acc],
                                           feed_dict={model.x1: np.expand_dims(s1s[i], axis=0),
                                                      model.x2: np.expand_dims(s2s[i], axis=0),
                                                      model.y: np.expand_dims(labels[i], axis=0),
                                                      model.features: np.expand_dims(features[i], axis=0)})
                MeanCost += c
                Accuracys.append(a)
                Sentences.append(pred)
                if i % 200 == 0:
                    print('[batch {}]  cost: {}  accuracy: {}'.format(i, c, a))
            print('Mean Cost: {}   Mean Accuracy: {}'.format(MeanCost/i, np.mean(Accuracys)))

    print("=" * 50)
    print("max accuracy: {}  mean accuracy: {}".format(max(Accuracys), np.mean(Accuracys)))
    print("=" * 50)
    print('Number Sentences: {}'.format(len(Sentences)))

    fasttext = gensim.models.KeyedVectors.load("wiki.dump")
    print('FastText loaded')
    with open('output.txt', 'w') as f:
        for sen in Sentences[:2]:
            string = ''
            for word in sen:
                string += fasttext.similar_by_vector(word)[0][0] + ' '
            string += '\n'
            f.write(string)


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
        "epoch": 50,
        "model_type": "End2End",
        "max_len": 40,
        "num_layers": 4,
        "data": 'OneEnglish',
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

