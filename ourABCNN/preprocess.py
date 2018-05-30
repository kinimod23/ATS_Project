import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import gensim
import random 
import codecs
import copy
import time


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


class Data():
    """
    def __init__(self, word2vec, max_len=0):
        self.s1s, self.s2s, self.labels, self.features = [], [], [], []
        self.index, self.max_len, self.word2vec = 0, max_len, word2vec
    """
    def __init__(self, max_len=0):
        self.s1s, self.s2s, self.labels, self.features = [], [], [], []
        self.index, self.max_len = 0, max_len
    
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
        s1_mats, s2_mats = [], []

        for i in range(batch_size):
            s1 = self.s1s[self.index + i]
            s2 = self.s2s[self.index + i]

            # [1, d0, s]
            s1_mats.append(np.expand_dims(np.pad(np.column_stack([self.word2vec.get(w) for w in s1]),
                                                 [[0, 0], [0, self.max_len - len(s1)]],
                                                 "constant"), axis=0))

            s2_mats.append(np.expand_dims(np.pad(np.column_stack([self.word2vec.get(w) for w in s2]),
                                                 [[0, 0], [0, self.max_len - len(s2)]],
                                                 "constant"), axis=0))

        # [batch_size, d0, s]
        batch_s1s = np.concatenate(s1_mats, axis=0)
        batch_s2s = np.concatenate(s2_mats, axis=0)
        batch_labels = self.labels[self.index:self.index + batch_size]
        batch_features = self.features[self.index:self.index + batch_size]

        self.index += batch_size

        return batch_s1s, batch_s2s, batch_labels, batch_features


class MSRP(Data):
    def open_file(self, mode, parsing_method="normal"):
        with open("./MSRP_Corpus/msr_paraphrase_" + mode + ".txt", "r", encoding="utf-8") as f:
            f.readline()

            for line in f:
                items = line[:-1].split("\t")
                label = int(items[0])
                if parsing_method == "NLTK":
                    s1 = nltk.word_tokenize(items[3])
                    s2 = nltk.word_tokenize(items[4])
                else:
                    s1 = items[3].strip().split()
                    s2 = items[4].strip().split()

                # bleu_score = nltk.translate.bleu_score.sentence_bleu(s1, s2)
                # sentence_bleu(s1, s2, smoothing_function=nltk.translate.bleu_score.SmoothingFunction.method1)

                self.s1s.append(s1)
                self.s2s.append(s2)
                self.labels.append(label)
                self.features.append([len(s1), len(s2)])

                # double use training data
                """
                if mode == "train":
                    self.s1s.append(s2)
                    self.s2s.append(s1)
                    self.labels.append(label)
                    self.features.append([len(s2), len(s1)])
                """

                local_max_len = max(len(s1), len(s2))
                if local_max_len > self.max_len:
                    self.max_len = local_max_len

        self.data_size = len(self.s1s)
        self.num_features = len(self.features[0])


class WikiQA(Data):
    def open_file(self, mode):
        with open("./WikiQA_Corpus/WikiQA-" + mode + ".txt", "r", encoding="utf-8") as f:
            stopwords = nltk.corpus.stopwords.words("english")

            for line in f:
                items = line[:-1].split("\t")

                s1 = items[0].lower().split()
                # truncate answers to 40 tokens.
                s2 = items[1].lower().split()[:40]
                label = int(items[2])

                self.s1s.append(s1)
                self.s2s.append(s2)
                self.labels.append(label)
                word_cnt = len([word for word in s1 if (word not in stopwords) and (word in s2)])
                self.features.append([len(s1), len(s2), word_cnt])

                local_max_len = max(len(s1), len(s2))
                if local_max_len > self.max_len:
                    self.max_len = local_max_len

        self.data_size = len(self.s1s)

        flatten = lambda l: [item for sublist in l for item in sublist]
        q_vocab = list(set(flatten(self.s1s)))
        idf = {}
        for w in q_vocab:
            idf[w] = np.log(self.data_size / len([1 for s1 in self.s1s if w in s1]))

        for i in range(self.data_size):
            wgt_word_cnt = sum([idf[word] for word in self.s1s[i] if (word not in stopwords) and (word in self.s2s[i])])
            self.features[i].append(wgt_word_cnt)

        self.num_features = len(self.features[0])

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
        while wrong_per_sen < 4:
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
    def open_file(self, mode, method): # mode = test, train etc only for file name
        with codecs.open("../corpus/complex_" + mode + ".txt", 'r', encoding="utf-8") as c, \
        codecs.open("../corpus/simple_" + mode + ".txt", 'r', encoding="utf-8") as s:   
            for(complex_sen, simple_sen) in zip(c.readlines(), s.readlines()):
                s1 = word_tokenize(complex_sen.strip().lower())
                s2 = word_tokenize(simple_sen.strip().lower())
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
            for s1 in self.s1s:
                set_s1.append(set(s1))
            print("set creation took {}s".format(time.time() - set_begin))
            idf_begin = time.time()
            for w in q_vocab:
                count = 0
                for s1 in set_s1:
                    if w in s1:
                        count += 1
                idf[w] = np.log(self.data_size / float(count))
            #idf[w] = np.log(self.data_size / len([1 for set(s1) in self.s1s if w in s1]))
            #idf = {w:np.log(self.data_size / len([1 for s1 in self.s1s if w in s1]))for w in q_vocab}
            print("Idf dict creation took: {}".format(time.time() - idf_begin))

            if method == "labeled":
                # create sentences that don't match, label them as 0 
                self.process_labeled()

            elif method == "unlabeled":
                # no labels needed, simple sentence in label 
                # to compare output and "label" with BLEU
                simple_sen =  []
                for line in s.readlines():
                    simple_sen.append(word_tokenize(line.strip().lower()))
                self.labels = simple_sen
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

if __name__ == '__main__':
    train_data = ComplexSimple()
    train_data.open_file(mode="train", method="labeled")
