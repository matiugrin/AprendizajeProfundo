# Trabajo Practico, Clasificación de texto del MeLi Challenge 2019.

De las pruebas realizadas, las dos siguientes son las que mayor tiempo se pudieron ejecutar (10 batch) en CPU.

### Se entreno una Red Convolucional con los siguientes parámetros: (/experiment/cnn.py)
    
--embeddings-size   300 \
--filters-lenght    2 3 4 5 \
--filter-count      100 \
--epochs            10

### Con el test de Validacion se obtuvieron los siguientes resultados: (En este caso no se utilizo el Set de Test)
train_loss:         1.08
validation_bacc:    0.738
validation_loss:    1.122

    self.fc = nn.Linear(filter_count * len(filters_lenght), 1024)
    self.output = nn.Linear(1024, n_labels)

### Segunda prueba de 10 Batch, el mayor cambio se realizo en el tamano del vector de features. (1024)

--embeddings-size   300 \
--filters-lenght    2 3 4 \
--filter-count      100 \
--epochs            10

### Con el test de Validacion se obtuvieron los siguientes resultados:
**test_bacc	        0.809**  
**test_loss	        0.817**  
train_loss	        1.015  
validation_bacc	    0.755  
validation_loss	    1.046  

    self.fc = nn.Linear(filter_count * len(filters_lenght), 1024)
    self.output = nn.Linear(1024, n_labels)


De los experimentos realizados, ese fue el mejor modelo obtenido, se hicieron otras pruebas con diferente cantidad de filtros y filters-lenght, pero solo se corrieron 2 epochs en cada una. 

### MLP
Por otra parte se realizaron pruebas con MLP (embeddings_size	300, epochs	5, hidden_layers	[256, 128]). Cuyos valores obtenidos son:
train_loss:2.409
validation_bacc:0.493
validation_loss:2.412


### MlFlow

En la carpeta mlruns se encuentran todas las pruebas realizadas tanto para CNN como para MLP para el set de datos en Español.



# Aprendizaje Profundo

Repositorio oficial de la materia optativa "Aprendizaje Profundo" (Deep Learning) de la Diplomatura en Ciencias de Datos de la UNC.

Para comenzar a instalar y configurar el entorno de trabajo por favor seguir las instrucciones detalladas en el [Notebook 0](./0_set_up.ipynb).
