import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, ElasticNetCV

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import f_regression, SelectKBest, SelectPercentile, SelectFromModel

from sklearn.model_selection import train_test_split, cross_val_score


todrop = ['uf10','sc51','sc53','sc54','sc110','sc111','sc112','sc113','sc115','sc116','sc117','uf5','sc125','uf6','sc127','uf7','sc134','uf7a','uf9','sc140','sc141','uf8','sc143','sc144','sc159','uf12','sc161','uf13','uf14','sc164','uf15','sc166','uf16','sc174','uf64','sc181','uf17a','sc548','sc549','sc550','sc551','sc570','sc574','sc560','uf53','uf54','uf19','rec1','uf46','rec4','rec_race_a','rec_race_c','rec28','tot_per','uf26','uf28','uf27','rec39','uf42','uf42a','uf34','uf34a','uf35','uf35a','uf36','uf36a','uf37','uf37a','uf38','uf38a','uf39','uf39a','uf40','uf40a','uf30','uf29','rec8','rec7','chufw','flg_sx1','flg_ag1','flg_hs1','flg_rc1','hflag2','hflag1','hflag13','hflag6','hflag3','hflag14','hflag16','hflag7','hflag9','hflag10','hflag91','hflag11','hflag12','hflag4','hflag18','uf52h_h','uf52h_a','uf52h_b','uf52h_c','uf52h_d','uf52h_e','uf52h_f','uf52h_g','sc541','sc184','sc542','sc543','sc544','hhr2','hhr5','race1','sc52','uf2a','uf2b','uf43','sc27']
continuous_cols = ['fw','uf11','sc150','sc151','sc186','sc571','uf23','rec54','rec53','seqno']
categorical_cols = ['sc118','sc114','new_csr','sc121','sc120','sc26','boro','sc23','sc37','sc173','sc152','sc153','sc154','sc155','sc156','sc157','sc158','sc197','sc189','sc193','sc196','sc199','rec15','rec62','rec64','cd','uf48']
boolean_cols = ['uf1_1','uf1_2','uf1_3','uf1_4','uf1_5','uf1_6','uf1_7','uf1_8','uf1_9','uf1_10','uf1_11','uf1_12','uf1_13','uf1_14','uf1_15','uf1_16','uf1_17','uf1_18','uf1_19','uf1_20','uf1_21','uf1_22','uf1_35','sc24', 'sc36','sc38','sc147','sc149','sc171','sc185', 'sc198', 'sc187', 'sc188', 'sc190', 'sc191', 'sc192', 'sc194', 'sc575','rec21']

na_map = {
    'sc23': [8],
    'sc38': [3,8],
    'sc173': [3,8],
    'sc154': [8],
    'sc157': [8],
    'sc197': [4,8],
    'sc189': [5,8],
    'sc196': [8],
    'sc199': [9],
    'rec15': [10, 11, 12],
    'uf1_1': [8],
    'uf1_2': [8],
    'uf1_3': [8],
    'uf1_4': [8],
    'uf1_5': [8],
    'uf1_6': [8],
    'uf1_7': [8],
    'uf1_8': [8],
    'uf1_9': [8],
    'uf1_10': [8],
    'uf1_11': [8],
    'uf1_12': [8],
    'uf1_13': [8],
    'uf1_14': [8],
    'uf1_15': [8],
    'uf1_16': [8],
    'uf1_17': [8],
    'uf1_18': [8],
    'uf1_19': [8],
    'uf1_20': [8],
    'uf1_21': [8],
    'uf1_22': [8],
    'uf1_35': [8],
    'sc24': [8],
    'sc36': [3, 8],
    'sc147': [3, 8],
    'sc171': [3, 8],
    'sc185': [8],
    'sc198': [8],
    'sc187': [8],
    'sc188': [8],
    'sc190': [8],
    'sc191': [8],
    'sc192': [8],
    'sc194': [8],
    'sc575': [3, 8],
    'rec21': [8],
    'sc186': [8],
    'sc571': [5,8],
    'rec54': [7],
    'rec53': [9],
    'sc26': [],
    'sc120': [5,8],
    'sc121': [3,8],
    'new_csr': [],
    'sc114': [4],
    'sc118': [8],
}

label = 'uf17'


def download_data():
    return pd.read_csv('https://ndownloader.figshare.com/files/7586326')

def drop_leaks(data):
    data_drop = data.drop(todrop, axis=1)
    print("Shape after drop leaks: ", data_drop.shape)
    return data_drop

def drop_missing_label(data):
    print("Before dropping missing labels: ", data.shape)
    data_drop = data[data[label].astype(float) != 99999.0]
    print("After dropping mising labels: ", data_drop.shape)
    return data_drop

def separate_features_labels(data):
    labels = data[label]
    data = data.drop(label, axis=1)
    return (data, labels)

def mark_missing_values(data):
    missing_counts = []
    for key in na_map:
        missing_for_this_col = data[data[key].isin(na_map[key])]
        count = len(missing_for_this_col)
        missing_counts.append((key, count))
    print("Counts of missing values: ", missing_counts)

    transformed_counts = []
    for key in na_map:
        mask = [data[key].isin(na_map[key])]
        data.iloc[mask, data.columns.get_loc(key)] = 'NaN'
        transformed_counts.append((key, 0))

    return data

def do_train_test_split(data, labels):
    return train_test_split(data, labels, random_state=42)

def do_impute(data):
    imp = Imputer(strategy="most_frequent").fit(data)
    data_imputed = imp.transform(data)
    data_imputed = pd.DataFrame(data_imputed, columns=data.columns)
    return data_imputed


dummy_categorical_cols = []

def do_dummies(data):
    print("Data before dummies: ", data.shape)
    data_dummies = pd.get_dummies(data, columns=categorical_cols)
    print("Data after dummies: ", data_dummies.shape)
    
    # Create list of new categorical dummy column names
    for c in data_dummies.columns:
        for d in categorical_cols:
                if d in c:
                    dummy_categorical_cols.append(c) 
    
    print("Dummy cols created: ", len(dummy_categorical_cols))
    return data_dummies

def do_feature_transformation(data):
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly = poly.fit(data)

    data_transformed = poly.transform(data)
    print("Transformed data: ", data_transformed.shape)

    return data_transformed

def do_feature_selection(x_train, x_test, y_train):
    from_model = SelectFromModel(Lasso(), threshold=0.05)
    from_model.fit(x_train, y_train)
    x_train_select = from_model.transform(x_train)
    x_test_select = from_model.transform(x_test)
    print("Selected x_train: ", x_train_select.shape)
    print("Selected x_test: ", x_test_select.shape)
    return (x_train_select, x_test_select)

def do_feature_scaling(x_train, x_test):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    print("Scaled features train: ", x_train_scaled.shape)
    print("Scaled features test: ", x_test_scaled.shape)
    return (x_train_scaled, x_test_scaled)

def do_cross_val(x_train, y_train):
    reg = RidgeCV()
    scores = cross_val_score(reg, x_train, y_train, cv=5)
    mean_cvs = np.mean(scores)
    return mean_cvs

def score_rent(x_train, x_test, y_train, y_test):
    reg = RidgeCV()
    reg.fit(x_train, y_train)
    score = reg.score(x_test, y_test)
    return score

def predict_rent(x_train, x_test, y_train, y_test):
    reg = RidgeCV()
    reg.fit(x_train, y_train)
    preds = reg.predict(x_test)
    return (np.array(x_test), np.array(y_test), np.array(preds))


if __name__ == '__main__':
    data = download_data()
    data = drop_leaks(data)
    data = drop_missing_label(data)
    data, labels = separate_features_labels(data)
    data = mark_missing_values(data)

    x_train, x_test, y_train, y_test = do_train_test_split(data, labels)

    x_train = do_impute(x_train)
    x_test = do_impute(x_test)

    x_train = do_dummies(x_train)
    x_test = do_dummies(x_test)

    x_train = do_feature_transformation(x_train)
    x_test = do_feature_transformation(x_test)

    x_train, x_test = do_feature_selection(x_train, x_test, y_train)
    x_train, x_test = do_feature_scaling(x_train, x_test)

    mean_cvs = do_cross_val(x_train, y_train)
    print("Mean CVS: ", mean_cvs)

    test_score = score_rent(x_train, x_test, y_train, y_test)
    print("Test score: ", test_score)

    x_test, y_test, preds = predict_rent(x_train, x_test, y_train, y_test)
    print("Mean prediction error: ", np.absolute(preds - y_test).mean())
