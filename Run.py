import os
import pandas as pd
from Model import Wide, Deep, Wide_Deep
from Data import Data

file_path = os.path.dirname(os.path.realpath('__file__'))

class Run:

    def __init__(self):

        columns = ["age", "workclass", "fnlwgt", "education", "education_num",
                   "marital_status", "occupation", "relationship", "race", "gender",
                   "capital_gain", "capital_loss", "hours_per_week", "native_country",
                   "income_bracket"]

        self.df_train = pd.read_csv(file_path + '/data/adult.data', sep=',', names = columns)
        self.df_test = pd.read_csv(file_path + '/data/adult.test', sep=',', names = columns, skipinitialspace=True, skiprows=1)

        # infome_label추가
        self.df_train['income_label'] = (self.df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
        self.df_test['income_label'] = (self.df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

        # age_group 추가
        age_groups = [0, 25, 65, 90]
        age_labels = range(len(age_groups) - 1)
        self.df_train['age_group'] = pd.cut(self.df_train['age'], age_groups, labels=age_labels)
        self.df_test['age_group'] = pd.cut(self.df_test['age'], age_groups, labels=age_labels)

        target = 'income_label'

    def Wide(self):

        load = Data()
        X_train, y_train, X_test, y_test = load.get_wide_model_data(self.df_train, self.df_test)

        model = Wide(X_train, y_train)
        model = model.get_model()
        model.fit(X_train, y_train, epochs=10,  batch_size=64)

        print('wide model accuracy:', model.evaluate(X_test, y_test)[1])

    def Deep(self):

        load = Data()
        X_train, y_train, X_test, y_test, \
        embeddings_tensors, continuous_tensors = load.get_deep_model_data(self.df_train, self.df_test)

        model = Deep(X_train, y_train, embeddings_tensors, continuous_tensors)
        model = model.get_model()
        model.fit(X_train, y_train, batch_size=64, epochs=10)

        print('deep model accuracy:', model.evaluate(X_test, y_test)[1])

    def Wide_and_Deep(self):

        load = Data()
        X_train_wide, y_train_wide, X_test_wide, y_test_wide = load.get_wide_model_data(self.df_train, self.df_test)
        X_train_deep, y_train_deep, X_test_deep, y_test_deep, \
        embeddings_tensors, continuous_tensors = load.get_deep_model_data(self.df_train, self.df_test)

        X_tr_wd = [X_train_wide] + X_train_deep
        y_tr_wd = y_train_deep

        X_te_wd = [X_test_wide] + X_test_deep
        y_te_wd = y_test_deep

        model = Wide_Deep(X_train_wide, y_train_wide, X_train_deep, y_train_deep, embeddings_tensors, continuous_tensors)
        model = model.get_model()
        model.fit(X_tr_wd, y_tr_wd, epochs=5, batch_size=128)

        print('wide and deep model accuracy:', model.evaluate(X_te_wd, y_te_wd)[1])

if __name__ == '__main__' :

    run = Run()
    run.Wide_and_Deep()

