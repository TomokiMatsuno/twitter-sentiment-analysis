from config import config
from neural import classifier

import preprocess
from preprocess import Vocab

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import numpy as np
import dynet as dy
import os

save_dir = '/saves/'

if not os.path.exists(os.curdir + save_dir):
    os.mkdir(os.curdir + save_dir)

# from gensim.test.utils import get_tmpfile

# embds = preprocess.Embeddings()
# embds.load()

path2data = '/Users/tomoki/NLP_data/sentiment-analysis-twitter/tweet-texts-segmented.txt'

texts = []
texts = []
with open(path2data, 'r') as f:
    for line in f.readlines():
        tokens = line.split('\t')
        texts.append((tokens[-1].strip('\n').split(), int(tokens[1])))

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]

fname = 'twitter_doc2vec_model_win5_d100'
# model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
# model = Doc2Vec(documents, window=5, min_count=3, workers=4, vector_size=100)

# fname = get_tmpfile("my_doc2vec_model")

# model.save(fname)
model = Doc2Vec.load(fname)


data = [[], []]
data_size = len(texts)
data[0], data[1] = texts[:-data_size // 10], texts[(data_size // 10) * 9:]

vocab = Vocab(data[0])
vocab.add_parsefile(data[1])


mdl = classifier.Model(word_dim=config.word_dim, hidden_dim=config.hidden_dim,
                  word_size=len(vocab._word_dict.x2i))

max_acc = 0
has_not_been_updated_for = 0

for epc in range(config.epochs):
    for step in range(2):
        losses = []
        tot_cor = 0
        tot_loss = 0
        isTrain = (1 - step)
        ids = [i for i in range(len(data[step]))]
        if isTrain:
            np.random.shuffle(ids)
        else:
            with open(os.curdir + save_dir + 'parsed.txt', 'w') as f:
                pass


        for i in ids:
            d = data[step][i]
            word_ids, gold_label = [vocab._word_dict.x2i[w] for w in d[0]], d[1]
            pred_label, loss = mdl.run(word_ids, gold_label, isTrain)
            losses.extend(loss)
            if isTrain:
                if len(losses) >= config.batch_size:
                    sum_loss = dy.esum(losses)
                    tot_loss += sum_loss.value()
                    sum_loss.backward()
                    mdl.update_parameters()
                    mdl._global_step += 1
                    losses = []
                    dy.renew_cg()

            else:
                if pred_label == gold_label:
                    tot_cor += 1

        if not isTrain:
            acc = tot_cor / len(data[step]) + 1e-10
            print('accuracy:', acc)
            if max_acc < acc:
                max_acc = acc
                has_not_been_updated_for = 0
                mdl._pc.save(os.curdir + save_dir + 'parameters')
            else:
                has_not_been_updated_for += 1
                if has_not_been_updated_for > config.quit_after_n_epochs_without_update:
                    print('accracy: ', max_acc)
                    exit(0)
        else:
            print('loss:', tot_loss)



