import tensorflow as tf
import numpy as np
import sys

from preprocess_dump2 import Word2Vec, ComplexSimple, FastText
from ABCNN_reduced import ABCNN
from utils import build_path
import os
import pickle
from time import time

def train(lr, w, l2_reg, epoch, model_type, data, word2vec, batch_size, num_layers, num_classes=2):

############################################################################
#########################   DATA LOADING   #################################
############################################################################
    if model_type == 'convolution': method = 'labeled'
    else: method = 'unlabeled'
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
    with tf.device("/gpu:0"):
        model = ABCNN(s=train_data.max_len, w=w, l2_reg=l2_reg, model_type=model_type,
                  num_features=train_data.num_features, num_classes=num_classes, num_layers=num_layers)

        optimizer = tf.train.AdagradOptimizer(lr, name="optimizer").minimize(model.cost)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=100)
        model_path = build_path("./models/", 'BCNN', num_layers, model_type, word2vec)

############################################################################
#########################     TRAINING     #################################
############################################################################
    with tf.Session(config=tfconfig) as sess:
        if model_type == 'deconvolution':
            saver.restore(sess, model_path + "-" + str(1000))
            print(model_path + "-" + str(1000), "restored.")
        train_summary_writer = tf.summary.FileWriter("../tf_logs/train2", sess.graph)
        sess.run(init)
        print("=" * 50)
        for e in range(1, epoch + 1):
            print("[Epoch " + str(e) + "]")
            train_data.reset_index()
            i , MeanCost, MeanAcc = 0, 0, 0
            while train_data.is_available():
                i += 1
                x1, x2, y, features = train_data.next_batch(batch_size=batch_size, model_type=model_type)
                merged, _, c, a = sess.run([model.merged, optimizer, model.cost, model.acc],
                                    feed_dict={model.x1: x1, model.x2: x2,
                                    model.y: y, model.features: features})
                MeanCost += c
                MeanAcc += a
                if i % 200 == 0:
                    print('[batch {}]  cost: {}  accuracy: {}'.format(i, c, a))
                train_summary_writer.add_summary(merged, i)
            print('Mean Cost: {}   Mean Accuracy: {}'.format(MeanCost/i, MeanAcc/i))
            if e % 100 == 0:
                save_path = saver.save(sess, build_path("./models/", data, 'BCNN', num_layers, model_type, word2vec), global_step=e)
                print("model saved as", save_path)
        print("training finished!")
        print("=" * 50)


if __name__ == "__main__":

    # Paramters
    # --lr: learning rate
    # --ws: window_size
    # --l2_reg: l2_reg modifier
    # --epoch: epoch
    # --batch_size: batch size
    # --model_type: model type
    # --num_layers: number of convolution layers
    # --data_type: MSRP or WikiQA data

    # default parameters
    params = {
        "lr": 0.08,
        "ws": 4,
        "l2_reg": 0.0004,
        "epoch": 50,
        "model_type": "End2End",
        "batch_size": 128,
        "num_layers": 4,
        "data": 'Wiki',
        "word2vec": 'FastText'
    }

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            k = arg.split("=")[0][2:]
            v = arg.split("=")[1]
            params[k] = v

    print("=" * 50)
    print("Parameters:")
    for k in sorted(params.keys()):
        print(k, ":", params[k])

    train(lr=float(params["lr"]), w=int(params["ws"]), l2_reg=float(params["l2_reg"]), epoch=int(params["epoch"]),
          model_type=params["model_type"], batch_size=int(params["batch_size"]), num_layers=int(params["num_layers"]),
            data=params["data"], word2vec=params["word2vec"])
