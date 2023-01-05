## Crapauduc

Dans ce fichier nous donnons une description de chaque notebook utilisé pour la mise en oeuvre de ce projet. 


###  [exploratory_analysis](../notebooks/eda_animals_analysis.ipynb)

Dans ce fichier a été fait l'analyse exploratoire des données. Elle se base sur les données contenues dans les fichiers [path_and_bounding_box](../data/path_and_bounding_box.csv) join avec celles de la [météo](../data/meteo.csv) sur la date. L'analyse s'est essentiellement basée sur les facteurs date heure et météo.

###  [meteo](../notebooks/eda_meteo.ipynb)
    
Dans ce fichier, les données météos ont été importées du format JSON dans lequel ils étaient à l'origine pour être mises sous le format .csv plus exploitable [météo](../data/meteo.csv). 

###  [example_bounding_box](../notebooks/example_bounding_box.ipynb)

Dans ce fichier nous pouvons voir une façon simple d'afficher la bounding box sur une image.

###  [filter_analysis](../notebooks/filter_analysis.ipynb)

Dans ce notebook est présenté la proportion d'images du dataset contenant des planches et celle n'en contenant pas. Pour se faire, ont été utilisées les fichiers .csv contenues dans [unnested_data](../data/unnested_data/). 

###  [filter_plank_application](../notebooks/filter_plank_application.ipynb)

Dans ce notebook sont étiquetés les images les labels [0,1] représentent la présence ou pas de planche,

###  [filter_plank_model](../notebooks/filter_plank_model.ipynb)
Dans ce notebook a été entrainé un modèle qui étant donnée une nouvelle image determine si elle contient une planche ou pas. Ceci nous sera utile dans la mesure où les animaux apparaissant sur des prises faites en dehors de la planche sont très peu visibles. 

###  [model_RCNN](../notebooks/model_RCNN.ipynb)

Ce notebook contient l'implémentation du model RCNN (Region-based CNN) adapté pour notre projet.

###  [model_SSD](../notebooks/model_SSD.ipynb)
 Ce model illustre une utilisation de [SSD(Single Shot Detector)](https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/). Le dataset d'entrainement doit être modifié pour qu'il puisse fournir des résultats sur celui ci.
