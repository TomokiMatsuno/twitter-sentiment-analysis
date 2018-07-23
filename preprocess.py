import codecs
import numpy as np

class Dict:
    def __init__(self, sents, initial_entries=None):
        self.i2x = {}
        self.x2i = {}
        self.appeared_x2i = {}
        self.appeared_i2x = {}
        self.freezed = False
        self.initial_entries = initial_entries

        if initial_entries is not None:
            for ent in initial_entries:
                self.add_entry(ent)

        for ent in sents:
            self.add_entry(ent)

        self.freeze()

    def add_entry(self, ent):
        if ent not in self.x2i:
            if not self.freezed:
                self.i2x[len(self.i2x)] = ent
                self.x2i[ent] = len(self.x2i)
            else:
                self.x2i[ent] = self.x2i['UNK']

    def add_entries(self, seq=None, minimal_count=0):
        if not self.freezed:
            for elem in seq:
                if self.cnt[elem] >= minimal_count and elem not in self.i2x:
                    self.i2x.append(elem)
            self.words_in_train = set(self.i2x)
        else:
            for ent in seq:
                if ent not in self.x2i:
                    self.x2i[ent] = self.x2i['UNK']


    def freeze(self):
        self.freezed = True

PATH2DATA = '/Users/tomoki/NLP_data/conll2018/task1/all/portuguese-dev'
PATH2TRAIN = ''
PATH2DEV = ''

class Vocab(object):
    def __init__(self, data):
        '''
        :param data:    A list of tuples (a list of strings (words in a sequence), int (label)).
        '''

        words = []
        for s in data:
            words.extend(s[0])

        self._word_dict = Dict(words, initial_entries=['<BOS>', '<EOS>', 'UNK'])

        return


    def add_parsefile(self, data):
        '''
        :param data:    A list of tuples (a list of strings (words in a sequence), int (label)).
        '''

        words = []
        for s in data:
            words.extend(s[0])

        self._word_dict.add_entries(words)

        return


class Embeddings(object):
    def __init__(self):
        self._x2i = dict()
        self._i2x = []
        self._embeddings_array = []
        self._embed_size = 0

    def load(self, file):
        with open(file, 'r') as f:
            cur_idx = 0
            embeddings = []
            reader = codecs.getreader('utf-8')(f, errors='ignore')
            for line_num, line in enumerate(reader):
                line = line.rstrip().split(' ')
                if len(line) > 1:
                    embeddings.append(np.array(line[1:], dtype=np.float32))
                    self._x2i[line[0]] = cur_idx
                    cur_idx += 1
            try:
                embeddings = np.stack(embeddings)
                embeddings = np.pad(embeddings, ((len(self.special_tokens), 0), (0, 0)), 'constant')
                self._embeddings_array = np.stack(embeddings)
                self._embed_size = self._embeddings_array.shape[1]
            except:
                shapes = set([embedding.shape for embedding in embeddings])
                raise ValueError("Couldn't stack embeddings with shapes in %s" % shapes)
        return

