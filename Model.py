import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2, l1_l2

class Wide:

    def __init__(self, X_train, y_train):

        wide_input = Input(shape=(X_train.shape[1],))  # (batch_size, X_train.shape[1])
        output_layer = Dense(y_train.shape[1], activation='sigmoid')(wide_input)

        # Model
        self.wide_model = Model(wide_input, output_layer)
        self.wide_model.compile(loss='binary_crossentropy', metrics= ['accuracy'], optimizer='Adam')

    def get_model(self):
        return self.wide_model

class Deep:

    def __init__(self, X_train, y_train, embeddings_tensors, continuous_tensors):

        deep_input = [et[0] for et in embeddings_tensors] + [ct[0] for ct in continuous_tensors]
        deep_embedding = [et[1] for et in embeddings_tensors] + [ct[1] for ct in continuous_tensors]
        deep_embedding = Flatten()(concatenate(deep_embedding))

        # layer 1
        layer_1 = Dense(100, activation='relu', kernel_regularizer= l1_l2(l1=0.01, l2=0.01))(deep_embedding)
        layer_1_dropout = Dropout(0.5)(layer_1)

        # layer 2
        layer_2 = Dense(50, activation='relu')(layer_1_dropout)
        layer_2_dropout = Dropout(0.5)(layer_2)

        # output
        output_layer = Dense(y_train.shape[1], activation='sigmoid')(layer_2_dropout)  # m = y_train.shape[1]

        # Model
        self.deep_model = Model(deep_input, output_layer)
        self.deep_model.compile(loss='binary_crossentropy', metrics= ['accuracy'], optimizer='Adam')

        # # 1개의 row에 대한 input 예시
        # x = [np.array([0.]),
        #      np.array([0.]),
        #      np.array([0.]),
        #      np.array([0.]),
        #      np.array([0.]),
        #      np.array([0.]),
        #      np.array([0.]),
        #      np.array([0.]),
        #      np.array([0.03067056]),
        #      np.array([0.1484529]),
        #      np.array([-0.21665953]),
        #      np.array([-0.03542945])]
        # y = y_train[0].reshape(-1,1)
        # deep_model.fit(x, y, batch_size=1, epochs=1)

    def get_model(self):
        return self.deep_model


class Wide_Deep:

    def __init__(self, X_train_wide, y_train_wide,
                       X_train_deep, y_train_deep,
                       embeddings_tensors, continuous_tensors):
        # WIDE
        wide_input = Input(shape=(X_train_wide.shape[1],), dtype='float32', name='wide')

        # DEEP
        deep_input = [et[0] for et in embeddings_tensors] + [ct[0] for ct in continuous_tensors]
        deep_embedding = [et[1] for et in embeddings_tensors] + [ct[1] for ct in continuous_tensors]

        deep_embedding = Flatten()(concatenate(deep_embedding))
        layer_1 = Dense(50, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(deep_embedding)
        layer_1_dropout = Dropout(0.5)(layer_1)
        layer_2 = Dense(20, activation='relu', name='deep')(layer_1_dropout)
        layer_2_dropout = Dropout(0.5)(layer_2)

        # WIDE & DEEP
        wd_input = concatenate([wide_input, layer_2_dropout])
        wd_output = Dense(y_train_deep.shape[1], activation='sigmoid', name='wide_deep')(wd_input)

        self.wide_deep_model = Model([wide_input, deep_input], wd_output)
        self.wide_deep_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    def get_model(self):
        return self.wide_deep_model







