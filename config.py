PCI = {
    #'pci_value' : [301, 302, 120, 154, 151, 448, 404, 433],
    'pci_value' : [37, 38, 39, 40, 41, 42, 120, 151, 154, 1, 62]
    #'pci_value' : [1, 2, 3, 54,151]
}

DNN = {
    'nb_epoch' : 100
}

FILE = {
    'path' : './data/51-5F/',
   # 'file_name' : 'pci_0917.csv'
   # 'file_name' : 'result_0608.csv'
   # 'file_name' : 'data_300_train_mod_1_train.csv'
    'file_name' : 'set4_125.csv'
}

MODEL = {
    'path' : './output/',
    'pci_model_name' : 'finalized_model_1-knn.sav',
    'rf_model_name' : 'finalized_model_2-knn.sav'  #rf radio map
}

GERNAL = {
    'train_data_size' : 50000,
    'nb_feature' : 2,
    'enb_feature_begin' : 8,
    'enb_feature_num' : 0
}
