
import numpy as np
import pandas as pd
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

class Data:

    def __init__(self):
        self.utils = Utils()

    def get_wide_model_data(self, df_train, df_test):

        df_train['IS_TRAIN'] = 1
        df_test['IS_TRAIN'] = 0
        df_wide = pd.concat([df_train, df_test])

        wide_cols = ['workclass', 'education', 'marital_status', 'occupation',
                     'relationship', 'race', 'gender', 'native_country', 'age_group']
        crossed_cols = (['education', 'occupation'], ['native_country', 'occupation'])
        target = 'income_label'
        categorical_columns = list(df_wide.select_dtypes(include=['object']).columns)

        crossed_columns_dic = self.utils.cross_columns(crossed_cols)
        wide_cols += list(crossed_columns_dic.keys())

        # crossed 변수 추가
        for col_name, col_lst in crossed_columns_dic.items():
            df_wide[col_name] = df_wide[col_lst].apply(lambda x: '-'.join(x), axis=1)
        df_wide = df_wide[wide_cols + [target] + ['IS_TRAIN']]

        # dummy로 변경
        dummy_cols = [wc for wc in wide_cols if wc in categorical_columns + list(crossed_columns_dic.keys())]
        df_wide = pd.get_dummies(df_wide, columns=dummy_cols)

        train = df_wide[df_wide.IS_TRAIN == 1].drop('IS_TRAIN', axis=1)
        test = df_wide[df_wide.IS_TRAIN == 0].drop('IS_TRAIN', axis=1)

        cols = [c for c in train.columns if c != target]
        X_train = train[cols].values
        X_train = np.array(X_train, dtype=np.float)
        y_train = train[target].values.reshape(-1, 1)

        X_test = test[cols].values
        X_test = np.array(X_test, dtype=np.float)
        y_test = test[target].values.reshape(-1, 1)

        return X_train, y_train, X_test, y_test

    def get_deep_model_data(self, df_train, df_test):

        df_train['IS_TRAIN'] = 1
        df_test['IS_TRAIN'] = 0
        df_deep = pd.concat([df_train, df_test])

        embedding_cols = ['workclass', 'education', 'marital_status', 'occupation',
                          'relationship', 'race', 'gender', 'native_country']
        cont_cols = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']
        target = 'income_label'

        deep_cols = embedding_cols + cont_cols
        df_deep = df_deep[deep_cols + [target, 'IS_TRAIN']]

        scaler = StandardScaler()
        df_deep[cont_cols] = pd.DataFrame(scaler.fit_transform(df_train[cont_cols]), columns=cont_cols)
        df_deep, col_to_unique_val_num = self.utils.val2idx(df_deep, embedding_cols)

        train = df_deep[df_deep.IS_TRAIN == 1].drop('IS_TRAIN', axis=1)
        test = df_deep[df_deep.IS_TRAIN == 0].drop('IS_TRAIN', axis=1)

        embeddings_tensors = []
        for ec in embedding_cols:
            layer_name = ec + '_inp'
            inp = Input(shape=(1,), dtype='int64', name=layer_name)
            embd = Embedding(col_to_unique_val_num[ec], 8, input_length=1, embeddings_regularizer=l2(1e-3))(inp)
            embeddings_tensors.append((inp, embd))
            del (inp, embd)

        continuous_tensors = []
        for cc in cont_cols:
            layer_name = cc + '_in'
            inp = Input(shape=(1,), dtype='float32', name=layer_name)
            bulid = Reshape((1, 1))(inp)
            continuous_tensors.append((inp, bulid))
            del (inp, bulid)

        X_train = [train[c] for c in deep_cols]
        y_train = np.array(train[target].values).reshape(-1, 1)

        X_test = [test[c] for c in deep_cols]
        y_test = np.array(test[target].values).reshape(-1, 1)

        return X_train, y_train, X_test, y_test, embeddings_tensors, continuous_tensors

class Utils:

    def cross_columns(self, x_cols):

        crossed_columns = dict()
        colnames = ['_'.join(x_c) for x_c in x_cols]
        for cname, x_c in zip(colnames, x_cols):
            crossed_columns[cname] = x_c
        return crossed_columns

    def val2idx(self, df, cols):

        val_types = dict()
        for c in cols:
            val_types[c] = df[c].unique()

        val_to_idx = dict()
        for k, v in val_types.items():
            val_to_idx[k] = {o: i for i, o in enumerate(val_types[k])}

        for k, v in val_to_idx.items():
            df[k] = df[k].apply(lambda x: v[x])

        unique_vals = dict()  # 사용한 값만
        for c in cols:
            unique_vals[c] = df[c].nunique()

        return df, unique_vals


