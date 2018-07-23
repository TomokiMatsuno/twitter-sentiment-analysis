import dynet as dy
import numpy as np
from config import config
from neural import utils
from preprocess import Vocab

BOS, EOS = 0, 1
NUM_CLASS = 5

class Model(object):
    def __init__(self, word_dim, hidden_dim, word_size):
        self._global_step = 0


        self._word_size = word_size
        self._word_dim = word_dim
        # self._doc_size = doc_size
        # self._doc_dim = doc_dim
        self._hidden_dim = hidden_dim

        self._pc = dy.ParameterCollection()

        if config.adam:
            self._trainer = dy.AdamTrainer(self._pc, config.learning_rate, config.beta_1, config.beta_2, config.epsilon)
        else:
            # self._trainer = dy.AdadeltaTrainer(self._pc)
            trainer = dy.SimpleSGDTrainer(self._pc, config.learning_rate)
            trainer.set_clip_threshold(config.clip_threshold)

        # self._trainer.set_clip_threshold(1.0)

        self.params = dict()

        self.lp_w = self._pc.add_lookup_parameters((self._word_size, self._word_dim))

        self._pdrop_embs = config.pdrop_embs
        self._pdrop_lstm = config.pdrop_lstm
        self._pdrop_mlp = config.pdrop_mlp

        self.LSTM_builders = []

        f = dy.VanillaLSTMBuilder(1, self._word_dim, hidden_dim, self._pc)
        b = dy.VanillaLSTMBuilder(1, self._word_dim, hidden_dim, self._pc)

        self.LSTM_builders.append((f, b))
        for i in range(config.layers - 1):
            f = dy.VanillaLSTMBuilder(1, 2 * hidden_dim, hidden_dim, self._pc)
            b = dy.VanillaLSTMBuilder(1, 2 * hidden_dim, hidden_dim, self._pc)
            self.LSTM_builders.append((f, b))

        self.dec_LSTM = dy.VanillaLSTMBuilder(config.layers, hidden_dim, hidden_dim, self._pc)
        #
        # f = dy.SimpleRNNBuilder(1, char_dim, hidden_dim, self._pc)
        # b = dy.SimpleRNNBuilder(1, char_dim, hidden_dim, self._pc)
        #
        # self.LSTM_builders.append((f, b))
        # for i in range(config.layers - 1):
        #     f = dy.SimpleRNNBuilder(1, 2 * hidden_dim, hidden_dim, self._pc)
        #     b = dy.SimpleRNNBuilder(1, 2 * hidden_dim, hidden_dim, self._pc)
        #     self.LSTM_builders.append((f, b))
        #
        #
        #
        # self.dec_LSTM = dy.SimpleRNNBuilder(1, hidden_dim, hidden_dim, self._pc)

        self.MLP = self._pc.add_parameters((hidden_dim, hidden_dim * 4))
        self.MLP_bias = self._pc.add_parameters((hidden_dim))
        self.classifier = self._pc.add_parameters((NUM_CLASS, hidden_dim))
        self.classifier_bias = self._pc.add_parameters((NUM_CLASS))
        self.MLP_attn = self._pc.add_parameters((hidden_dim * 2, hidden_dim))
        self.MLP_attn_bias = self._pc.add_parameters((hidden_dim * 2))
        self.attn_weight = self._pc.add_parameters((hidden_dim * 2))

    def run(self, s, t, isTrain):
        pred = -1
        losses = []

        MLP = dy.parameter(self.MLP)
        MLP_bias = dy.parameter(self.MLP_bias)
        MLP_attn = dy.parameter(self.MLP_attn)
        MLP_attn_bias = dy.parameter(self.MLP_attn_bias)
        attn_weight = dy.parameter(self.attn_weight)
        classifier = dy.parameter(self.classifier)
        classifier_bias = dy.parameter(self.classifier_bias)

        s = [BOS] + s + [EOS]
        word_embs = [self.lp_w[w] for w in s]
        top_recur = utils.biLSTM(self.LSTM_builders, word_embs,
                                 dropout_h=self._pdrop_lstm if isTrain else 0.,
                                 dropout_x=self._pdrop_lstm if isTrain else 0.)
        key = dy.concatenate_cols(top_recur[1:-1])

        h = dy.concatenate([top_recur[0], top_recur[-1]])

        h = dy.affine_transform([MLP_bias, MLP, h])

        if isTrain:
            h = dy.dropout(h, self._pdrop_mlp)

        h = dy.rectify(h)
        score = dy.affine_transform([classifier_bias, classifier, h])

        if isTrain:
            losses.append(dy.pickneglogsoftmax(score, t))
        else:
            pred = score.npvalue().argmax()

        return pred, losses







        # feat_embs = []
        # for idx in range(len(self.lp_feats)):
        #     if idx < len(f):
        #         feat_embs.append(self.lp_feats[idx][f[idx]])
        #     else:
        #         feat_embs.append(dy.inputVector(np.zeros(self._feat_dim)))
        # feat_embs = dy.concatenate(feat_embs)
        #
        # prev_char = BOS
        pred_word = []
        losses = []

        prev_top_recur = dy.inputVector(np.zeros(self._hidden_dim))
        state = self.dec_LSTM.initial_state()
        idx = 0



        while prev_char != EOW:
            # feat_embs = dy.cube(feat_embs)

            tmp = dy.concatenate([self.lp_c[prev_char], feat_embs, prev_top_recur])

            if isTrain:
                tmp = dy.dropout(tmp, self._pdrop_embs)

            h = dy.affine_transform([MLP_attn_bias, MLP_attn, tmp])
            if isTrain:
                h = dy.dropout(h, self._pdrop_mlp)

            query = dy.cmult(attn_weight, dy.rectify(h))
            attn_vec = dy.softmax(dy.transpose(key) * query)
            value = key * attn_vec
            inp = dy.concatenate([value, tmp])
            inp = dy.affine_transform([MLP_bias, MLP, inp])
            h = state.add_input(inp).output()
            top_recur = dy.rectify(h)
            if isTrain:
                top_recur = dy.dropout(top_recur, self._pdrop_mlp)
            prev_top_recur = h
            score = dy.affine_transform([classifier_bias, classifier, top_recur])
            if isTrain:
                losses.append(dy.pickneglogsoftmax(score, t[idx + 1]))
                prev_char = t[idx + 1]
                idx += 1
            else:
                pred_char = score.npvalue().argmax()
                pred_word.append(pred_char)
                prev_char = pred_char
                if len(pred_word) > 30:
                    break

        return pred_word, losses

    def update_parameters(self):
        self._trainer.learning_rate = config.learning_rate * config.decay ** (self._global_step / config.decay_steps)
        self._trainer.update()
