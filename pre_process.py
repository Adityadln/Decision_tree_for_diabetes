from imports import *
def preprocess_data(data, target_column):
    X = data.drop(columns=target_column)
    y = data[target_column]
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    X = pd.DataFrame(X, columns=data.columns.drop(target_column))

    columns_to_normalize = X.columns.difference([target_column])

    scaler = StandardScaler()
    X[columns_to_normalize] = scaler.fit_transform(X[columns_to_normalize])
    X[target_column] = y.reset_index(drop=True) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    X_train[target_column] = X_train[target_column].astype(int)
    X_test[target_column] = X_test[target_column].astype(int)
    return X_train, X_test, y_train, y_test