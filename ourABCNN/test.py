import tensorflow as tf
import numpy as np
import sys

from preprocess import Word2Vec, ComplexSimple, FastText
from BCNN import ABCNN_conv, ABCNN_deconv
from utils import build_path
import pickle
import os
import gensim
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def test(w, l2_reg, epoch, max_len, model_type, data, word2vec, num_layers, num_classes=2):

############################################################################
#########################   DATA LOADING   #################################
############################################################################
    method = 'labeled'
    dumped = 'preprocessed_train_'+method+'_'+data+'_'+word2vec+'.pkl'
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

        if model_type != 'convolution':
            decoder = ABCNN_deconv(lr=0.08, s=test_data.max_len, w=w, l2_reg=l2_reg,
                      num_layers=num_layers)

        model_path = build_path("./models/", data, 'BCNN', num_layers, model_type, word2vec)
        model_path_old = build_path("./models/", data, 'BCNN', 3, 'convolution', word2vec)

        variables_enc = tf.trainable_variables(scope='Encoder')
        enc_saver = tf.train.Saver(var_list = variables_enc, max_to_keep=2)

        if model_type != 'convolution':
            variables_dec = tf.trainable_variables(scope='Decoder')
            dec_saver = tf.train.Saver(var_list = variables_dec, max_to_keep=2)

        if model_type == 'convolution':
            nc_saver.restore(sess, model_path_old + "-" + str(1000))
        elif model_type == 'deconvolution':
            enc_saver.restore(sess, model_path_old + "-" + str(1000))
            dec_saver.restore(sess, model_path + "-" + str(170))
        else:
            enc_saver.restore(sess, model_path + "-" + str(240))
            dec_saver.restore(sess, model_path + "-" + str(240))

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
        for i in range(50):

            if model_type == 'convolution':
                pred, c2, a2 = sess.run([encoder.prediction, encoder.cost, encoder.acc],
                                       feed_dict={encoder.x1: np.expand_dims(s1s[i], axis=0),
                                                  encoder.x2: np.expand_dims(s2s[i], axis=0),
                                                  encoder.y1: np.expand_dims(labels[i], axis=0)})
                print(pred.shape)
            elif model_type == 'deconvolution':
                pred, c1, a1 = sess.run([encoder.prediction, encoder.cost, encoder.acc],
                                       feed_dict={encoder.x1: np.expand_dims(s1s[i], axis=0),
                                                  encoder.x2: np.expand_dims(s2s[i], axis=0),
                                                  encoder.y1: np.expand_dims(labels[i], axis=0)})
                output, c2, a2 = sess.run([decoder.prediction, decoder.cost, decoder.acc],
                                        feed_dict={encoder.x1: np.expand_dims(s1s[i], axis=0),
                                                  encoder.x2: np.expand_dims(s2s[i], axis=0),
                                                  encoder.y1: np.expand_dims(labels[i], axis=0),
                                                  decoder.x: pred,
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
            for sen in Sentences:
                string = ''
                for word in range(40):
                    string += fasttext.wv.similar_by_vector(sen[0,:,word], topn=1)[0][0] + ' '
                string += '\n'
                f.write(string)
        print('Output created!')

if __name__ == "__main__":

    # default parameters
    params = {
        "ws": 4,
        "l2_reg": 0.0004,
        "epoch": 1,
        "model_type": "End2End",
        "max_len": 50,
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

