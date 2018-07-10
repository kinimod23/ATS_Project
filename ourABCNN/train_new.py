import tensorflow as tf
import numpy as np
import sys

from preprocess_dump import MSRP, WikiQA
from preprocess_dump2 import Word2Vec, ComplexSimple
from ABCNN_reduced import ABCNN
from utils import build_path
from sklearn import linear_model, svm
from sklearn.externals import joblib
import os
import pickle
from time import time

def train(lr, w, l2_reg, epoch, model_type, batch_size, num_layers, data_type, method, word2vec, dumped_data, num_classes=2):
    if data_type == "WikiQA":
        train_data = WikiQA(word2vec=word2vec)
        train_data.open_file(mode="train")
    elif data_type == 'Paraphrase':
        train_data = MSRP(word2vec=word2vec)
        train_data.open_file(mode="train")
    elif data_type == 'Complex2Simple':
        dumped = dumped_data+'_train_'+method+'.pkl'
        if not os.path.exists(dumped):
            print("Dumped data not found! Data will be preprocessed")
            train_data = ComplexSimple(word2vec=word2vec)
            train_data.open_file(mode="train", method=method)
        else:
            print("found pickled state, loading..")
            train_data = ComplexSimple(word2vec=word2vec)
            with open(dumped, 'rb') as f:
                dump_dict = pickle.load(f)
                for k, v in dump_dict.items():
                    setattr(train_data, k, v)
            print("done!")

    print("=" * 50)
    print("training data size:", train_data.data_size)
    print("training max len:", train_data.max_len)
    print("=" * 50)

    tfconfig = tf.ConfigProto(allow_soft_placement = True)
    with tf.device("/gpu:0"):
        model = ABCNN(s=train_data.max_len, w=w, l2_reg=l2_reg, model_type=model_type,
                  num_features=train_data.num_features, num_classes=num_classes, num_layers=num_layers)

        optimizer = tf.train.AdagradOptimizer(lr, name="optimizer").minimize(model.cost)

        # Due to GTX 970 memory issues
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

        # Initialize all variables
        init = tf.global_variables_initializer()

        # model(parameters) saver
        saver = tf.train.Saver(max_to_keep=100)
        model_path = build_path("./models/", 'BCNN', num_layers, model_type)

    #with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session(config=tfconfig) as sess:
        train_summary_writer = tf.summary.FileWriter("../tf_logs/train2", sess.graph)

        sess.run(init)

        print("=" * 50)
        for e in range(1, epoch + 1):
            print("[Epoch " + str(e) + "]")

            train_data.reset_index()
            i = 0
            MeanCost = 0
            batchTime = 0
            modelTime = 0

            if model_type == 'deconvolution':
                saver.restore(sess, model_path + "-" + str(100))
                print(model_path + "-" + str(100), "restored.")

            #LR = linear_model.LogisticRegression()
            #SVM = svm.LinearSVC()
            #clf_features = []

            while train_data.is_available():
                i += 1
                startBatch = time()
                batch_x1, batch_x2, batch_y, batch_features = train_data.next_batch(batch_size=batch_size, model_type=model_type)
                batchTime += time()-startBatch

                startModel = time()
                merged, _, c, features = sess.run([model.merged, optimizer, model.cost, model.output_features],
                                                  feed_dict={model.x1: batch_x1,
                                                             model.x2: batch_x2,
                                                             model.y: batch_y,
                                                             model.features: batch_features})
                modelTime += time()-startModel
                MeanCost += c

                #clf_features.append(features)

                if i % 100 == 0:
                    print('[batch {}]  cost: {}'.format(i, c, ))
                train_summary_writer.add_summary(merged, i)
            print('Mean Cost: {}'.format(MeanCost/i))

            if e % 100 == 0:
                save_path = saver.save(sess, build_path("./models/", 'BCNN', num_layers, model_type), global_step=e)
                print("model saved as", save_path)



            #if model_type == 'convolution':
            #    clf_features = np.concatenate(clf_features)
            #    LR.fit(clf_features, train_data.labels)
            #    SVM.fit(clf_features, train_data.labels)
#
            #    if e % 50 == 0:
            #        LR_path = build_path("./models/", data_type, 'BCNN', num_layers, "-" + str(e) + "-LR.pkl")
            #        SVM_path = build_path("./models/", data_type, 'BCNN', num_layers, "-" + str(e) + "-SVM.pkl")
            #        joblib.dump(LR, LR_path)
            #        joblib.dump(SVM, SVM_path)
#
            #        print("LR saved as", LR_path)
            #        print("SVM saved as", SVM_path)

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
        "data_type": "Complex2Simple",
        "dumped_data": "preprocessed_train.pkl",
        "method": "unlabeled",
        "word2vec": Word2Vec()
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
          data_type=params["data_type"], method=params["method"], word2vec=params["word2vec"], dumped_data=params["dumped_data"])
