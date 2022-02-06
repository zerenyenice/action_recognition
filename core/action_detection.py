import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.autograd as autograd
import torch.nn as nn


class ActionDetection:
    def __init__(self, model_path):
        self.angle_idx = [[2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13]]
        self.dist_idx = [3, 4, 6, 7, 9, 10, 12, 13]
        self.action_dict = {0: 'Standing', 1: 'Walking', 2: 'Hitting', 3: 'Kicking'}
        self.model = LSTMClassifier(64,4)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def calculate_features(self, body_points):

        features = []
        count = 0
        for bp_i in body_points:

            angles = [self.get_angle(bp_i[i], bp_i[j]) for i, j in self.angle_idx]
            if count == 0:
                bp_i_t = bp_i
                angles_t = np.array(angles)

            delta_angle = abs((np.array(angles) - (angles_t))).tolist()
            delta_dist = [self.get_dist(bp_i_t[j], bp_i[j]) for j in self.dist_idx]

            features.append(angles + delta_angle + delta_dist)
            bp_i_t = bp_i
            angles_t = np.array(angles)
            count = count + 1

        features = np.array(features)
        features = np.concatenate([np.zeros(((features.shape[0] // 32 + 1) * 32 - features.shape[0], 24)), features])
        features = np.expand_dims(features, 0)

        return torch.from_numpy(features).float()

    def recognize_actions(self, body_points):

        data = self.calculate_features(body_points)
        out = []
        with torch.no_grad():
            if torch.cuda.is_available():
                for i in range(data.shape[1]):
                    out.append(self.model(data[:, i:i + 32, :].cuda()).argmax(-1).cpu().numpy()[0])
            else:
                for i in range(data.shape[1]):
                    out.append(self.model(data[:, i:i + 32, :]).argmax(-1).numpy()[0])

        return out

    @staticmethod
    def get_dist(joint_1, joint_2):
        x1, y1 = joint_1
        x2, y2 = joint_2
        if (x1 == 0) or (x2 == 0) or (y1 == 0):
            return 0.0
        else:
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @staticmethod
    def get_angle(joint_1, joint_2):
        x1, y1 = joint_1
        x2, y2 = joint_2

        if (x1 == 0) or (x2 == 0):
            return 0.0
        dx = x2 - x1
        dy = y2 - y1
        rad = math.atan2(dy, dx)
        degree = (rad * 180) / math.pi
        if (degree < 0):
            degree = degree + 360
        return degree


"""
class ARTrainer:
    import tensorflow as tf
    def __init__(self, val_ratio=0.1):
        self.val_ratio = val_ratio
        self.train = pd.read_csv('data/pose_36.txt', index_col=None, header=None)
        self.out = pd.read_csv('data/pose36_c.txt', index_col=None, header=None)

    def create_data(self):
        out = pd.get_dummies(self.out[0])
        train = self.train.values
        train = train.reshape(-1, 32, 24)
        print(train.shape)
        out = out.values

        out = np.expand_dims(out, -1).transpose(0, 2, 1)
        out = np.repeat(out, 32, axis=1)
        print(out.shape)

        train_inp, val_inp, train_out, val_out = train_test_split(train, out, test_size=self.val_ratio, shuffle=True)

        return train_inp, val_inp, train_out, val_out

    @classmethod
    def create_tf_model(cls):
        series_input = cls.tf.keras.layers.Input((32, 24))
        series_input = cls.tf.keras.layers.GaussianNoise(0.05)(series_input)
        # x = cls.tf.keras.layers.Dropout(0.25)(series_input)
        x = cls.tf.keras.layers.LSTM(64, dropout=0.2, return_sequences=True)(series_input)
        x = cls.tf.keras.layers.TimeDistributed(cls.tf.keras.layers.Dropout(0.2))(x)
        x = cls.tf.keras.layers.TimeDistributed(cls.tf.keras.layers.Dense(16, activation='relu'))(x)
        x = cls.tf.keras.layers.TimeDistributed(cls.tf.keras.layers.Dropout(0.05))(x)
        x = cls.tf.keras.layers.TimeDistributed(cls.tf.keras.layers.Dense(4))(x)
        x = cls.tf.keras.layers.TimeDistributed(cls.tf.keras.layers.Softmax())(x)

        return cls.tf.keras.Model(series_input, x)

    @classmethod
    def compile_tf_model(cls, m):
        optimizer = cls.tf.optimizers.Adam(learning_rate=0.001)
        m.compile(optimizer=optimizer, loss=cls.tf.keras.losses.categorical_crossentropy,
                  metrics=cls.tf.keras.metrics.categorical_accuracy)
        return m

    @classmethod
    def create_tf_checkpoint(cls):
        ckpt = cls.tf.keras.callbacks.ModelCheckpoint(
            'model/check_weights.h5', monitor='val_categorical_accuracy', verbose=0, save_best_only=True, mode='min',
            save_weights_only=True
        )
        return ckpt

    def train_tf_model(self, batch_size):
        train_inp, val_inp, train_out, val_out = self.create_data()
        model = self.create_tf_model()
        model = self.compile_tf_model(model)
        checkpoints = self.create_tf_checkpoint()

        model.fit(x=train_inp, y=train_out, batch_size=batch_size, epochs=250, validation_batch_size=batch_size,
                  validation_data=(val_inp, val_out), callbacks=[checkpoints], shuffle=True)

        model.save('model/classifier.h5', include_optimizer=False)
        model.save_weights('model/classifier_w.h5')

        return model

    def create_tensorflow_pytorh_model(self):
        tf_model = self.train_tf_model(128)
        weights = tf_model.get_weights()
        pt_model = LSTMClassifier(64, 4)

        pt_model.lstm.weight_ih_l0.data = torch.from_numpy(weights[0]).transpose(1, 0).contiguous()
        pt_model.lstm.weight_hh_l0.data = torch.from_numpy(weights[1]).transpose(1, 0).contiguous()

        pt_model.lstm.bias_ih_l0.data = torch.from_numpy(weights[2]).contiguous()
        pt_model.lstm.bias_hh_l0.data = torch.from_numpy(weights[2]).contiguous()

        pt_model.hidden2inter.weight.data = torch.from_numpy(weights[3]).transpose(1, 0).contiguous()
        pt_model.hidden2inter.bias.data = torch.from_numpy(weights[4]).contiguous()
        pt_model.hidden2out.weight.data = torch.from_numpy(weights[5]).transpose(1, 0).contiguous()
        pt_model.hidden2out.bias.data = torch.from_numpy(weights[6]).contiguous()

        torch.save(pt_model, 'model/torch_classifier.h5')
        torch.save(pt_model.state_dict(), 'torch_state.h5')

"""


class LSTMClassifier(nn.Module):

    def __init__(self, hidden_dim, output_size):
        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(24, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)

        self.hidden2inter = nn.Linear(hidden_dim, 16)
        self.act1 = nn.ReLU()
        self.hidden2out = nn.Linear(16, 4)
        self.softmax = nn.LogSoftmax()

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
                autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))

    def forward(self, batch):
        self.hidden = self.init_hidden(batch.size(-1))

        output, (ht, ct) = self.lstm(batch)
        # output = output.contiguous().view(-1, self.hidden_dim)
        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        output = self.hidden2inter(ht[-1])  # torch.cat((ht[0], ht[1]), dim=-1))
        output = self.act1(output)
        output = self.hidden2out(output)
        output = self.softmax(output)
        # output = output.contiguous().view(batch.size(0),-1, 4)

        return output

