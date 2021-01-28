from Experiments import *
from DataLoader import DataLoader


if __name__ == '__main__':

    # load data
    dl_1 = DataLoader('data\\UCI-bank-marketing.csv', 'outputs\\Marketing', 'Marketing')
    dl_1.load_data()
    # run classifier
    ANN(dl_1)
    BOOST(dl_1)
    SVM_RBF(dl_1)
    SVM_linear(dl_1)
    KNN(dl_1)
    DT(dl_1)

    # load data
    dl_2 = DataLoader('data\\Heart.csv', 'outputs\\Heart', 'Heart')
    dl_2.load_data()
    # run classifier
    ANN(dl_2)
    BOOST(dl_2)
    SVM_linear(dl_2)
    KNN(dl_2)
    SVM_RBF(dl_2)
    DT(dl_2)

    # # load data
    # dl_3 = DataLoader('data\\Cancer.csv', 'outputs\\Cancer', 'Cancer')
    # dl_3.load_data()
    # # run classifier
    # ANN(dl_3)
    # BOOST(dl_3)
    # SVM_RBF(dl_3)
    # SVM_linear(dl_3)
    # KNN(dl_3)
    # DT(dl_3)
