import os
import urllib.request
from pathlib import Path
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def download(url, dataset):
    """ Descargar un dataset desde la web

    Args:
        url (str): url del sitio de descarga del dataset
        dataset (str): nombre del dataset a descargar
    """
    def fn(foo):
        def gn(*args, **kwargs):
            dir_path = os.path.join(Path(__file__).parent, "data")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            urllib.request.urlretrieve(
                url + dataset + ".data", data := os.path.join(dir_path, dataset + ".data"))
            urllib.request.urlretrieve(
                url + dataset + ".names", name := os.path.join(dir_path, dataset + ".names"))
            return foo(data, name)
        return gn
    return fn


def stratified_split(X, y, test_size, random_state=42):
    """ Aplicar un stratified split al dataset 

    Args:
        X (Pandas.DataFrame): dataset
        y (Pandas.DataFrame): targets
        test_size (float): proporción de la partición de prueba
        random_state (int, opcional): aleatoriedad de los índices de entrenamiento y prueba producidos. Default en 42.

    Returns:
        Pandas.DataFrame: partición de entrenamiento
        Pandas.DataFrame: partición de prueba
        Numpy.ndarray: targets de entrenamiento
        Numpy.ndarray: targets de prueba
    """
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    split.get_n_splits(X, y)
    for train_idx, test_idx in split.split(X, y):
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    return X_train, X_test, y_train.squeeze().ravel(), y_test.squeeze().ravel()


class Appending_Attributes(BaseEstimator, TransformerMixin):
    """Custom Transformer para el pipeline de procesamiento con el fin de
    proponer nuevos feactures en el dataset
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Z = X.copy()
        # se generan nuevos feactures de acuerdo al análisis
        # de correlación realizado
        Z["A7/A6"] = Z["A7"] / Z["A6"]  # Flavanoids / Total phenols
        # Flavanoids / OD280/OD315 of diluted wines
        Z["A7/A12"] = Z["A7"] / Z["A12"]
        Z["A7/A9"] = Z["A7"] / Z["A9"]  # Flavanoids / Proanthocyanins
        Z["A7/A11"] = Z["A7"] / Z["A11"]  # Flavanoids / Hue
        Z["A7/A13"] = Z["A7"] / Z["A13"]  # Flavanoids / Proline
        Z["A2/A7"] = Z["A2"] / Z["A7"]  # Malic acid / Flavanoids
        Z["A8/A7"] = Z["A8"] / Z["A7"]  # Nonflavanoid phenols / Flavanoids
        return Z


@download(url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine/", dataset="wine")
def retreive_dataset(data, name):
    """[summary]

    Args:
        data (str): dirección en donde se ubica el archivo .data
                    correspondiente al dataset descargado
        name (str): dirección en donde se ubica el archivo .name
                    correspondiente al dataset descargado

    Returns:
        Pandas.DataFrame: dataset
        Pandas.DataFrame: targets
    """
    # se genera un Pandas.DataFrame con el dataset y se proporciona un identificador
    # a cada atributo:
    #
    # A1)  Alcohol
    # A2)  Malic acid
    # A3)  Ash
    # A4)  Alcalinity of ash
    # A5)  Magnesium
    # A6)  Total phenols
    # A7)  Flavanoids
    # A8)  Nonflavanoid phenols
    # A9)  Proanthocyanins
    # A10) Color intensity
    # A11) Hue
    # A12) OD280/OD315 of diluted wines
    # A13) Proline
    wines = pd.read_csv(data, names=['CLASS', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6',
                                     'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13'])
    # se extraen los targets del dataset original
    y = pd.DataFrame(wines['CLASS'], columns=['CLASS'])
    # también se aislan los atributos
    X = wines.drop('CLASS', axis='columns')
    return X, y


def preprocessing(foo):
    """ Preprocesar los datos según el análisis previo

    Args:
        foo (function): función a la cual se le proporcionarán los
                        datos preprocesados
    """
    def fn(*args, **kwargs):
        # generando el dataset a partir de los datos descargados
        X, y = retreive_dataset()
        # separación del dataset en train y test
        X_train, X_test, y_train, y_test = stratified_split(
            X, y, test_size=0.2)
        # definimos un pipeline para el preprocesamiento
        pipe = Pipeline([
            ('appending_attributes', Appending_Attributes()),
            ('scale', StandardScaler()),
        ])
        # aplicamos el pipeline para ambas particiones del dataset
        X_train = pipe.fit_transform(X_train)
        X_test = pipe.transform(X_test)
        # datos listos para ser implementados en un modelo
        foo(X_train, X_test, y_train, y_test)
    return fn
