# Trabajo Practico, Clasificación de texto del MeLi Challenge 2019.

### Se entreno una Red Convolucional con los siguientes parametros:
    
--embeddings-size 300 \
--filters-lenght 2 3 4 5 \
--filter-count 100 \
--epochs 10

### Con el test de Validacion se obtuvieron los siguientes resultados:
train_loss:1.08
validation_bacc:0.738
validation_loss:1.122

Ese fue el mejor modelo obtenido, se hicieron pruebas con diferente cantidad de filtros, epochs y filters-lenght. 

Por otra parte se realizaron pruebas con MLP (embeddings_size	300, epochs	5, hidden_layers	[256, 128]). Cuyos valores obtenidos son:
train_loss:2.409
validation_bacc:0.493
validation_loss:2.412




# Aprendizaje Profundo

Repositorio oficial de la materia optativa "Aprendizaje Profundo" (Deep Learning) de la Diplomatura en Ciencias de Datos de la UNC.

Para comenzar a instalar y configurar el entorno de trabajo por favor seguir las instrucciones detalladas en el [Notebook 0](./0_set_up.ipynb).
