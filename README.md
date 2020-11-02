# Wines-with-SVM
En la siguiente práctica se implementaron el algoritmo SVM en la clasificación de 178 vinos italianos de tres cultivares de los cuales mediante análisis químico se determinaron 13 características.

## Descripción del dataset
Los datos analizados provienen del repositorio de aprendizaje automático _wine_. Este conjunto de datos se utiliza a menudo para probar y comparar el rendimiento de varios algoritmos de clasificación. Un análisis químico de 178 vinos italianos de tres cultivares diferentes arrojó 13 características, las cuales se describen a continuación: 

* __Ácido málico:__ Es un tipo de ácido con fuerte acidez y aroma a manzana. El vino tinto se acompaña naturalmente de ácido málico.
* __Ceniza:__ La esencia de la ceniza es una sal inorgánica, que tiene un efecto sobre el sabor general del vino y puede darle una sensación de frescura.
* Alcalinidad de la ceniza: Es una medida de alcalinidad débil disuelta en agua.
* __Magnesio:__ es un elemento esencial del cuerpo humano, que puede promover el metabolismo energético y es débilmente alcalino.
* __Fenoles totales:__ moléculas que contienen sustancias polifenólicas, que tienen un sabor amargo y afectan el sabor, color y sabor del vino, y pertenecen a los nutrientes del vino.
* __Flavonoides:__ Es un antioxidante beneficioso para el corazón y anti-envejecimiento, rico en aroma y amargo.
* __Fenoles no flavonoides:__ es un gas aromático especial con resistencia a la oxidación y es débilmente ácido.
* __Proantocianinas:__ es un compuesto bioflavonoide, que también es un antioxidante natural con un ligero olor amargo.
* __Intensidad del color:__ se refiere al grado de matiz del color. Se utiliza para medir el estilo del vino para que sea "ligero" o "espeso". La intensidad del color es alta, mientras que cuanto más tiempo estén en contacto el vino y el mosto durante el proceso de elaboración del vino, más espeso será el sabor.
* __Matiz:__ se refiere a la intensidad del color y al grado de calidez y frialdad. Puede utilizarse para medir la variedad y la edad del vino. Los vinos tintos con edades más altas tendrán un tono amarillo y una mayor transparencia. La intensidad y el tono del color son indicadores importantes para evaluar la calidad del aspecto de un vino.
* __Prolina:__ es el principal aminoácido del vino tinto y una parte importante de la nutrición y el sabor del vino.
* __DO280/DO315 de vinos diluidos:__ este es un método para determinar la concentración de proteínas, que puede determinar el contenido de proteínas de varios vinos.

## Extrayendo el dataset
De primera instancia se procedió a descargar el dataset [wine](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/) y por medio de la implementación de la librería _Pandas_ a clasificar y analizar los datos para su posterior preprocesamiento:
```Python
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
```
A continuación se puede observar el ```DataFrame``` generado:

![](/images/X.png)

## Analizando los datos
Una vez extraido y preparado el dataset (vease test.py) se calculó la matriz de correlación:

![](/images/attribute-correlations.png)

en la cual se puede observar que existen correlaciones entre ciertos datos, lo cual nos puede ser de interés a la hora de llevar nuesto dataset a un algoritmo de clasificación. Corroboramos numéricamente algunas de estas correlaciones:

```Python
    print(corr_matrix['A7'].sort_values(ascending=False))
    # A7   1.000000
    # A6   0.864564
    # A12  0.787194
    # A9   0.652692
    # A11  0.543479
    # A13  0.494193
    # A1   0.236815
    # A5   0.195784
    # A3   0.115077
    # A10 -0.172379
    # A4  -0.351370
    # A2  -0.411007
    # A8  -0.537900
    # Name: A7, dtype: float64
```

y de igual forma de manera gráfica:

![](/images/correlations.png)

podemos afirmar que existe una relación directa entre la cantidad de flavonoides y:
* los fenoles totales
* la concentración de proteínas
* las proantocianinas
* el matiz
* la prolina

presentes en los vinos analizados. A su vez, existe una relación inversa entre los flavonoides y:
* el ácido málico
* los fenoles no flavonoides

## Preparando el _pipeline_

Una vez analizado el dataset y haber identificado correlaciones entre las características, se procede a construir un pipeline que realice el preprocesamiento.

```Python
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
```
Mediante la función ```retreive_dataset``` cargamos el dataset original y después realizamos un _stratified split_ para separar nuestros datos en entrenamiento y prueba, a una relación 80%-20% respectivamente.

![](/images/stratified-split-histogram.png)

Una vez realizado el split definimos nuestro pipeline donde primero agregaremos las características propuestas:

![](/images/Z.png)

después nomalizaremos los datos mediante un _standar scaler_, teniendo al final del _pipeline_ un dataset preprocesado con las características aumentadas y escalado.

## Aplicando algoritmos de clasificación

Ya preprocesados los datos, se procede a implementar dos algoritmos de clasificación, a saber: un __Support Vector Machine SVM__ y un __Multilayer Perceptron MLP__, ambos  algoritmos que tomarán una copia de los datos de entrenamiento y generarán una clasificación para los datos de prueba. Cada implementación hará uso de un _cross validation_, el cual dividirá el set de entrenamiento en particiones que servirán para entrenar de mejor forma al modelo, y un _grid search_ para buscar, de una serie de parámetros requeridos para cada algoritmo, la configuración de los mismos que den como resultado un mejor entrenamiento:

### SVM

```Python
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
```

### MLP

```Python
    # generamos un MLP con gridsearch para la selección de hyperparámetros y un cross validation que dividirá
    # en 10 secciones nuestra partición de entrenamiento, tomando cada una como una partición de validación
    model = GridSearchCV(MLPClassifier(), {'solver': ['sgd', 'adam'], 'activation': ['logistic', 'tanh', 'relu'], 'max_iter': [
                         2000, 5000, 10000], 'hidden_layer_sizes': (142, 20)}, cv=10, verbose=1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
```

## Resultados

Se ejecutaron ambos algoritmos y se mostraron la mejor configuración de parámetros para ambos, así como el desempeño del _cross validation_ y el desempeño de los algoritmos con los datos de prueba:

```
    Fitting 10 folds for each of 12 candidates, totalling 120 fits
    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 120 out of 120 | elapsed:    0.2s finished
    Best C: 10
    Best Kernel: rbf
    Best Gamma: 0.001
                  precision    recall  f1-score   support

               1       1.00      1.00      1.00        12
               2       1.00      1.00      1.00        14
               3       1.00      1.00      1.00        10

        accuracy                           1.00        36
       macro avg       1.00      1.00      1.00        36
    weighted avg       1.00      1.00      1.00        36
```

### MLP

```
    Fitting 10 folds for each of 36 candidates, totalling 360 fits
    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 360 out of 360 | elapsed:  3.3min finished
    Best solver: sgd
    Best activation: logistic
    Best max_iter: 2000
                  precision    recall  f1-score   support

               1       0.92      1.00      0.96        12
               2       0.93      0.93      0.93        14
               3       1.00      0.90      0.95        10

        accuracy                           0.94        36
       macro avg       0.95      0.94      0.95        36
    weighted avg       0.95      0.94      0.94        36
```

## Referencias
Blake, C.L. and Merz, C.J. (1998), UCI Repository of machine learning databases, http://www.ics.uci.edu/~mlearn/MLRepository.html. Irvine, CA: University of California, Department of Information and Computer Science.

Bai, X., Wang, L., & Li, H. (2019). Identification of red wine categories based on physicochemical properties.