import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import gensim
import random
import codecs
import copy
import time
import os
from tqdm import tqdm
import _pickle as pickle

STATE_FN_SCHEME = "preprocessed_{}_{}_{}_{}.pkl"

class Word2Vec():
    def __init__(self):
        # Load Google's pre-trained Word2Vec model.
        self.model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',
                                                                     binary=True)
        self.unknowns = np.random.uniform(-0.01, 0.01, 300).astype("float32")

    def get(self, word):
        if word not in self.model.vocab:
            return self.unknowns
        else:
            return self.model.word_vec(word)

    def cntUnknowns(self, sentence, threshold):
        cnt = 0
        threshold_amount = len(sentence) * threshold
        for word in sentence:
            if word not in self.model.vocab:
                cnt += 1
        if cnt < threshold_amount:
            return True
        return False

class FastText():
    def __init__(self):
    # facebook AI's fasttext model trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens).
    # https://fasttext.cc/docs/en/english-vectors.html
        self.model = gensim.models.KeyedVectors.load("wiki.dump")
        self.unknowns = np.random.uniform(-0.01, 0.01, 300).astype("float32")

    def get(self,word):
        try:
            return self.model[word]
        except KeyError:
            return self.unknowns

    def cntUnknowns(self,sentence,threshold):
        cnt = 0
        threshold_amount = len(sentence) * threshold
        for word in sentence:
            try:
                a = self.model[word]
            except KeyError:
                cnt +=1
        if cnt < threshold_amount:
            return True
        return False

class Data():
    def __init__(self, word2vec, max_len=50):
        self.s1s, self.s2s, self.labels, self.features = [], [], [], []
        self.index, self.max_len, self.word2vec = 0, max_len, word2vec
    """
    def __init__(self, max_len=0):
        self.s1s, self.s2s, self.labels, self.features = [], [], [], []
        self.index, self.max_len = 0, max_len
    """
    def open_file(self):
        pass

    def is_available(self):
        if self.index < self.data_size:
            return True
        else:
            return False

    def reset_index(self):
        self.index = 0

    def next(self):
        if (self.is_available()):
            self.index += 1
            return self.data[self.index - 1]
        else:
            return

    def next_batch(self, batch_size):
        batch_size = min(self.data_size - self.index, batch_size)
        s1, s2, label_mats = [], [], []

        for i in range(batch_size):
            s1.append(self.s1_mats[self.index + i])
            s2.append(self.s2_mats[self.index + i])


        # [batch_size, d0, s]
        batch_s1s = np.concatenate(s1, axis=0)
        batch_s2s = np.concatenate(s2, axis=0)
        batch_labels = np.asarray(self.labels[self.index:self.index + batch_size], np.float32)

        self.index += batch_size

        return batch_s1s, batch_s2s, batch_labels


class ComplexSimple(Data):
    @staticmethod
    def diff(comp, sim):
        threshold = len(comp) / 3 # one third of the complex sen, as that should be longer
        comp = set(comp)
        in_common = len([item for item in sim if item in comp])
        if in_common <= threshold:
            return True
        return False

    def process_labeled(self):
        complex_sen = copy.copy(self.s1s)
        simple_sen =  copy.copy(self.s2s)

        wrong_per_sen = 0
        begin = time.time()
        while wrong_per_sen < 1:
            print("iter: {}".format(wrong_per_sen))
            rands = []
            for i in range(len(complex_sen)):
                rands.append(random.randint(0,len(complex_sen) - 1))
            for (num, comp) in zip(rands, complex_sen):
                if self.diff(simple_sen[num], comp):
                    self.s1s.append(comp)
                    self.s2s.append(simple_sen[num])
                    self.labels.append(0)
                else:
                    while True:
                        randi = random.randint(0, len(complex_sen) - 1)
                        if self.diff(simple_sen[randi], comp):
                            self.s1s.append(comp)
                            self.s2s.append(simple_sen[num])
                            self.labels.append(0)
                            break
            wrong_per_sen += 1
        took = time.time() - begin
        print("Took: {}".format(took))


    def open_file(self, mode, method, data, word2vec): # mode = test, train etc only for file name
        state_fn = STATE_FN_SCHEME.format(mode, method, data, word2vec)
        print("reading data...")
        if data == 'Wiki' :
            l1 = "../corpus/wiki_complex_" + mode + ".txt"
            l2 = "../corpus/wiki_simple_" + mode + ".txt"
        else:
            l1 = "../corpus/one_stop_complex" + ".txt"
            l2 = "../corpus/one_stop_simple" + ".txt"
        with codecs.open(l1, 'r', encoding="utf-8") as c, \
        codecs.open(l2, 'r', encoding="utf-8") as s:
            for(complex_sen, simple_sen) in tqdm(zip(c.readlines(), s.readlines())):
                s1 = word_tokenize(complex_sen.strip().lower())
                s2 = word_tokenize(simple_sen.strip().lower())
                if  not len(s1) > 40 and not len(s2) > 40:
                    if self.word2vec.cntUnknowns(s1, 0.1) and self.word2vec.cntUnknowns(s2, 0.1):
                        self.s1s.append(s1)
                        self.s2s.append(s2)
                        if method == "labeled":
                            self.labels.append(1)
            print("Data was read")

            self.data_size = len(self.s1s)

            flatten_begin = time.time()
            flatten = lambda l: [item for sublist in l for item in sublist]
            q_vocab = list(set(flatten(self.s1s)))
            print("flatten took: {}".format(time.time() - flatten_begin))
            idf = {}
            set_begin = time.time()
            set_s1 = []
            for s1 in tqdm(self.s1s):
                set_s1.append(set(s1))
            print("set creation took {}s".format(time.time() - set_begin))
            idf_begin = time.time()
            for w in tqdm(q_vocab):
                count = 0
                for s1 in set_s1:
                    if w in s1:
                        count += 1
                idf[w] = np.log(self.data_size / float(count))


            print("Idf dict creation took: {}".format(time.time() - idf_begin))

            if method == "labeled":
                # create sentences that don't match, label them as 0
                self.process_labeled()

            elif method == "unlabeled":
                # no labels needed, simple sentence in label
                # to compare output and "label" with BLEU
                self.labels = self.s2s
            else:
                raise NameError(method)


            # create features
            feature_begin = time.time()
            for (s1, s2) in zip(self.s1s, self.s2s):
                word_cnt = len([word for word in s1 if word in s2])
                self.features.append([len(s1), len(s2), word_cnt])
                local_max_len = max(len(s1), len(s2))
                if local_max_len > self.max_len:
                    self.max_len = local_max_len

            self.data_size = len(self.s1s)

            idf_feature_begin = time.time()
            for i in range(self.data_size):
                wgt_word_cnt = sum([idf[word] for word in self.s1s[i] if (word in self.s2s[i])])
                self.features[i].append(wgt_word_cnt)
            print("idf feature creation took: {}".format(time.time() - idf_feature_begin))
            self.num_features = len(self.features[0])
            feature_took = time.time() - feature_begin
            print("features took: {}".format(feature_took))


            self.s1_mats, self.s2_mats, self.labels_mats = [], [], []

            random.seed(4)
            random.shuffle(self.s1s)
            random.seed(4)
            random.shuffle(self.s2s)
            random.seed(4)
            random.shuffle(self.labels)
            random.seed(4)
            random.shuffle(self.features)

            word2vec_begin = time.time()
            for i in range(len(self.s1s)):
                self.s1_mats.append(np.expand_dims(np.pad(np.column_stack([self.word2vec.get(w) for w in self.s1s[i]]),
                                                 [[0, 0], [0, self.max_len - len(self.s1s[i])]],
                                                 "constant"), axis=0))

                self.s2_mats.append(np.expand_dims(np.pad(np.column_stack([self.word2vec.get(w) for w in self.s2s[i]]),
                                                 [[0, 0], [0, self.max_len - len(self.s2s[i])]],
                                                 "constant"), axis=0))
            print("word2vec took: {}".format(time.time()-word2vec_begin))

            p = pickle.Pickler(open(state_fn,"wb"))
            p.fast = True
            pickle_blacklist = [ "word2vec" ]
            dump_dict = dict()
            for k, v in self.__dict__.items():
                if k not in pickle_blacklist:
                    dump_dict[k] = v
            p.fast = True
            p.dump(dump_dict)


if __name__ == '__main__':
    mode = "50"
    if not os.path.exists(STATE_FN_SCHEME%mode):
        train_data = ComplexSimple(Word2Vec())
        train_data.open_file(mode=mode, method="unlabeled")
    else:
        print("found pickled state, loading..")
        train_data = ComplexSimple(Word2Vec())
        with open(STATE_FN_SCHEME%mode, "rb") as f:
            dump_dict = pickle.load(f)
            for k, v in dump_dict.items():
                setattr(train_data, k, v)

    print("done!")
