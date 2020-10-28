from wines import preprocessing
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


@preprocessing
def SVM_model(X_train, X_test, y_train, y_test):
    # generamos un SVC con gridsearch para la selección de hyperparámetros
    # y un cross validation que dividirá en 10 secciones nuestra partición
    # de entrenamiento, tomando cada una como una partición de validación
    model = GridSearchCV(SVC(), [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ], cv=10, verbose=1)
    # entrenamos nuestro modelo y generamos una predicción
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # observamos los hyperparámetros seleccionados y los resultados del
    # entrenamiento
    print('Best C:', model.best_estimator_.C)
    print('Best Kernel:', model.best_estimator_.kernel)
    print('Best Gamma:', model.best_estimator_.gamma)
    print(classification_report(y_test, predictions))


@preprocessing
def MLP_model(X_train, X_test, y_train, y_test):
    # generamos un MLP con gridsearch para la selección de hyperparámetros y un cross validation que dividirá
    # en 10 secciones nuestra partición de entrenamiento, tomando cada una como una partición de validación
    model = GridSearchCV(MLPClassifier(), {'solver': ['sgd', 'adam'], 'activation': ['logistic', 'tanh', 'relu'], 'max_iter': [
                         2000, 5000, 10000], 'hidden_layer_sizes': (142, 20)}, cv=10, verbose=1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # observamos los hyperparámetros seleccionados y los resultados del entrenamiento
    print('Best solver:', model.best_estimator_.solver)
    print('Best activation:', model.best_estimator_.activation)
    print('Best max_iter:', model.best_estimator_.max_iter)
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    # SVM_model()
    # Fitting 10 folds for each of 12 candidates, totalling 120 fits
    # [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    # [Parallel(n_jobs=1)]: Done 120 out of 120 | elapsed:    0.2s finished
    # Best C: 10
    # Best Kernel: rbf
    # Best Gamma: 0.001
    #               precision    recall  f1-score   support

    #            1       1.00      1.00      1.00        12
    #            2       1.00      1.00      1.00        14
    #            3       1.00      1.00      1.00        10

    #     accuracy                           1.00        36
    #    macro avg       1.00      1.00      1.00        36
    # weighted avg       1.00      1.00      1.00        36

    MLP_model()
    # Fitting 10 folds for each of 36 candidates, totalling 360 fits
    # [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    # [Parallel(n_jobs=1)]: Done 360 out of 360 | elapsed:  3.3min finished
    # Best solver: sgd
    # Best activation: logistic
    # Best max_iter: 2000
    #               precision    recall  f1-score   support

    #            1       0.92      1.00      0.96        12
    #            2       0.93      0.93      0.93        14
    #            3       1.00      0.90      0.95        10

    #     accuracy                           0.94        36
    #    macro avg       0.95      0.94      0.95        36
    # weighted avg       0.95      0.94      0.94        36
