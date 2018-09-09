import tensorflow as tf
import numpy as np
import sys

from preprocess_dump2 import Word2Vec, ComplexSimple, FastText
from ABCNN_splitted import ABCNN_conv, ABCNN_deconv
from utils import build_path
import pickle
import os
import gensim
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def test(w, l2_reg, epoch, max_len, model_type, data, word2vec, num_layers, num_classes=2):

############################################################################
#########################   DATA LOADING   #################################
############################################################################
    if model_type == 'convolution' or model_type == 'deconvolution': method = 'labeled'
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
    sess = tf.Session(config=tfconfig)

    with tf.device("/gpu:0"):

        encoder = ABCNN_conv(lr= 0.08, s=test_data.max_len, w=w, l2_reg=l2_reg,
                  num_layers=num_layers)

        saver = tf.train.Saver(max_to_keep=2)
        model_path = build_path("./models/", data, 'BCNN', num_layers, model_type, word2vec)
        model_path_old = build_path("./models/", data, 'BCNN', num_layers, 'convolution', word2vec)

        print('Before:')
        print(sess.run(tf.report_uninitialized_variables()))
        if model_type == 'deconvolution':
            saver.restore(sess, model_path_old + "-" + str(1000))
            print(model_path_old + "-" + str(1000), "restored.")
        print('Middle:')
        print(sess.run(tf.report_uninitialized_variables()))

        if model_type != 'convolution':
            decoder = ABCNN_deconv(lr=0.08, s=test_data.max_len, w=w, l2_reg=l2_reg,
                      num_layers=num_layers)
            saver.restore(sess, model_path + "-" + str(1000))
            print(model_path + "-" + str(1000), "restored.")
    print('After:')
    print(sess.run(tf.report_uninitialized_variables()))


############################################################################
#########################     TRAINING     #################################
############################################################################
    Accuracys = []
    Sentences = []
    for e in range(1, epoch + 1):
        print("[Epoch " + str(e) + "]")
        test_data.reset_index()
        i , MeanCost, MeanAcc, MeanEncAcc = 0, 0, 0, 0
        s1s, s2s, labels = test_data.next_batch(batch_size=test_data.data_size)
        for i in range(test_data.data_size):

            if model_type == 'convolution':
                pred, c2, a2 = sess.run([encoder.prediction, encoder.cost, encoder.acc],
                                       feed_dict={encoder.x1: np.expand_dims(s1s[i], axis=0),
                                                  encoder.x2: np.expand_dims(s2s[i], axis=0),
                                                  encoder.y1: np.expand_dims(labels[i], axis=0)})
            elif model_type == 'deconvolution':
                pred, c1, a1 = sess.run([encoder.prediction, encoder.cost, encoder.acc],
                                       feed_dict={encoder.x1: np.expand_dims(s1s[i], axis=0),
                                                  encoder.x2: np.expand_dims(s2s[i], axis=0),
                                                  encoder.y1: np.expand_dims(labels[i], axis=0)})
                output, c2, a2 = sess.run([decoder.prediction, decoder.cost, decoder.acc],
                                        feed_dict={encoder.x1: np.expand_dims(s1s[i], axis=0),
                                                  encoder.x2: np.expand_dims(s2s[i], axis=0),
                                                  encoder.y1: np.expand_dims(labels[i], axis=0),
                                                  decoder.x: np.expand_dims(pred[i], axis=0),
                                                  decoder.y: np.expand_dims(s2s[i], axis=0)})
                Sentences.append(output)
                MeanEncAcc += c1

            MeanCost += c2
            Accuracys.append(a2)
        print('Mean Cost: {}   Mean Accuracy: {}'.format(MeanCost/i, np.mean(Accuracys)))

    print("=" * 50)
    print("max accuracy: {}  mean accuracy: {}".format(max(Accuracys), np.mean(Accuracys)))
    print("=" * 50)
    print('Number of Sentences: {}'.format(len(Sentences)))

    if model_type != 'convolution':
        fasttext = gensim.models.KeyedVectors.load("wiki.dump")
        print('FastText loaded')
        with open('output.txt', 'w') as f:
            for sen in Sentences[:2]:
                string = ''
                for word in range(50):
                    string += fasttext.wv.similar_by_vector(sen[:,word], topn=1)[0] + ' '
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

