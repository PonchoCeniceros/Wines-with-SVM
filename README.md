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
## Analizando los datos
Una vez extraido y preparado el dataset (vease test.py) se calculó la matriz de correlación:

![](/images/attribute-correlations.png)

en la cual se puede observar que, tal y como se muestra en las siguientes gráficas

![](/images/correlations.png)

existe una relación directa entre la cantidad de flavonoides y:
* los fenoles totales
* la concentración de proteínas
* las proantocianinas
* el matiz
* la prolina

presentes en los vinos analizados. A su vez, existe una relación inversa entre los flavonoides y:
* el ácido málico
* los fenoles no flavonoides

## Preparando _pipeline_

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

Una vez realizado el split definimos nuestro pipeline donde primero agregaremos las características propuestas y nomalizaremos los datos con un _standar scaler_. De esta forma pasamos a tener un dataset con la forma:

![](/images/X.png)

A un dataset preprocesado con las características aumentadas y escalado:

![](/images/Z.png)

## Referencias
Blake, C.L. and Merz, C.J. (1998), UCI Repository of machine learning databases, http://www.ics.uci.edu/~mlearn/MLRepository.html. Irvine, CA: University of California, Department of Information and Computer Science.

Bai, X., Wang, L., & Li, H. (2019). Identification of red wine categories based on physicochemical properties.