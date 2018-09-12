import os
import tensorflow as tf
import numpy as np
import sys

from preprocess import Word2Vec, ComplexSimple, FastText
from BCNN import ABCNN_conv, ABCNN_deconv
from utils import build_path
import pickle
from time import time
import gensim
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train(lr, w, l2_reg, epoch, model_type, data, word2vec, batch_size, num_layers, num_classes=2):

############################################################################
#########################   DATA LOADING   #################################
############################################################################
    method = 'labeled'
    dumped = 'preprocessed_train_'+method+'_'+data+'_'+word2vec+'.pkl'
    if word2vec == 'FastText': w2v = FastText()
    else: w2v = Word2Vec()

    if not os.path.exists(dumped):
        print("Dumped data not found! Data will be preprocessed")
        train_data = ComplexSimple(word2vec=w2v)
        train_data.open_file(mode="train", method=method, data=data, word2vec=word2vec)
    else:
        print("found pickled state, loading..")
        train_data = ComplexSimple(word2vec=w2v)
        with open(dumped, 'rb') as f:
            dump_dict = pickle.load(f)
            for k, v in dump_dict.items():
                setattr(train_data, k, v)
        print("done!")

    print("=" * 50)
    print("training data size:", train_data.data_size)
    print("training max len:", train_data.max_len)
    print("=" * 50)

############################################################################
#########################      MODEL      ##################################
############################################################################
    tfconfig = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.Session(config=tfconfig)

    with tf.device("/gpu:0"):

        encoder = ABCNN_conv(lr=lr, s=train_data.max_len, w=w, l2_reg=l2_reg,
                  num_layers=num_layers)

        if model_type != 'convolution':
            decoder = ABCNN_deconv(lr=lr, s=train_data.max_len, w=w, l2_reg=l2_reg,
                      num_layers=num_layers)

        opt = tf.train.AdamOptimizer(lr, name="optimizer")
        optimizer_enc = opt.minimize(encoder.cost)
        variables_enc = tf.trainable_variables(scope='Encoder')
        enc_saver = tf.train.Saver(var_list = variables_enc, max_to_keep=2)

        if model_type != 'convolution':
            optimizer_dec = opt.minimize(decoder.cost, var_list=tf.trainable_variables(scope='Decoder'), name='opt_minimize')
            variables_dec = tf.trainable_variables(scope='Decoder') + opt.variables()
            dec_saver = tf.train.Saver(var_list = variables_dec, max_to_keep=2)
        model_path = build_path("./models/", data, 'BCNN', num_layers, model_type, word2vec)
        model_path_old = build_path("./models/", data, 'BCNN', num_layers, 'convolution', word2vec)

        if model_type == 'convolution':
            init = tf.variables_initializer(variables_enc + opt.variables())
        elif model_type == 'deconvolution':
            enc_saver.restore(sess, model_path_old + "-" + str(1000))
            print(model_path_old + "-" + str(1000), "restored.")
            init = tf.variables_initializer(variables_dec)
        else:
            init = tf.variables_initializer(tf.global_variables())


############################################################################
#########################     TRAINING     #################################
############################################################################

    train_summary_writer = tf.summary.FileWriter("../tf_logs/train2", sess.graph)
    sess.run(init)

    print("=" * 50)
    for e in range(1, epoch + 1):
        Sentences = []
        print("[Epoch " + str(e) + "]")
        train_data.reset_index()
        i , MeanCost, MeanAcc, MeanEncAcc = 0, 0, 0, 0
        while train_data.is_available():
            i += 1
            x1, x2, y = train_data.next_batch(batch_size=batch_size)
            if model_type == 'convolution':
                merged, _, c, a = sess.run([encoder.merged, optimizer_enc, encoder.cost, encoder.acc],
                                  feed_dict={encoder.x1: x1, encoder.x2: x2, encoder.y1: y})
            elif model_type == 'deconvolution':
                preds, acc_enc  = sess.run([encoder.prediction, encoder.acc],
                                  feed_dict={encoder.x1: x1, encoder.x2: x2, encoder.y1: y})
                merged, output, _, c, a = sess.run([decoder.merged, decoder.prediction, optimizer_dec, decoder.cost, decoder.acc],
                                  feed_dict={encoder.x1: x1, encoder.x2: x2, encoder.y1: y, decoder.x: preds, decoder.y: x2})
                Sentences.append(output)
                MeanEncAcc += acc_enc
            else:
                preds, _, acc_enc = sess.run([encoder.prediction, optimizer_enc, encoder.acc],
                                  feed_dict={encoder.x1: x1, encoder.x2: x2, encoder.y1: y})
                merged, output, _, c, a, a2 = sess.run([decoder.merged, decoder.prediction, optimizer_dec, decoder.cost, decoder.acc, encoder.acc],
                                  feed_dict={encoder.x1: x1, encoder.x2: x2, encoder.y1: y, decoder.x: preds, decoder.y: x2})
                Sentences.extend(output)
                MeanEncAcc += a2

            MeanCost += c
            MeanAcc += a

        train_summary_writer.add_summary(merged, i)
        print('Mean Encoder Accuracy: {:1.4f} Mean Cost: {:1.4f}   Mean Accuracy: {:1.4f}'.format(MeanEncAcc/i, MeanCost/i, MeanAcc/i))
        if e % 10 == 0:
            if model_type == 'convolution':
                save_path = enc_saver.save(sess, build_path("./models/", data, 'BCNN', num_layers, model_type, word2vec), global_step=e)
            elif model_type == 'deconvolution':
                save_path = dec_saver.save(sess, build_path("./models/", data, 'BCNN', num_layers, model_type, word2vec), global_step=e)
            else:
                enc_saver.save(sess, build_path("./models/", data, 'BCNN', num_layers, model_type, word2vec), global_step=e)
                dec_saver.save(sess, build_path("./models/", data, 'BCNN', num_layers, model_type, word2vec), global_step=e)
            print("model saved as", save_path)
    print("training finished!")
    print("=" * 50)
    if model_type != 'convolution':
        fasttext = gensim.models.KeyedVectors.load("wiki.dump")
        print('FastText loaded')
        with open('output.txt', 'w') as f:
            for sen in Sentences[-100:]:
                string = ''
                for word in range(50):
                    string += fasttext.wv.similar_by_vector(sen[0,:,word], topn=1)[0][0] + ' '
                string += '\n'
                f.write(string)
        print('Output created!')


if __name__ == "__main__":

    # default parameters
    params = {
        "lr": 0.08,
        "ws": 4,
        "l2_reg": 0.0004,
        "epoch": 100,
        "model_type": "End2End",
        "batch_size": 128,
        "num_layers": 2,
        "data": 'Wiki',
        "word2vec": 'FastText'
    }

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            k = arg.split("=")[0][2:]
            v = arg.split("=")[1]
            params[k] = v

    for i in range(3):
        print('\n')
    print("=" * 50)
    print("Parameters:")
    for k in sorted(params.keys()):
        print(k, ":", params[k])

    train(lr=float(params["lr"]), w=int(params["ws"]), l2_reg=float(params["l2_reg"]), epoch=int(params["epoch"]),
          model_type=params["model_type"], batch_size=int(params["batch_size"]), num_layers=int(params["num_layers"]),
            data=params["data"], word2vec=params["word2vec"])
