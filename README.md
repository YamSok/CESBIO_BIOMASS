# CESBIO_BIOMASS
Repo dédié au projet des méthodes de corrélation croisée adaptées à l'optimisation des corrections topographiques en imagerie radar à synthèse d'ouverture : applications aux images aéroportées bande P pour la mission BIOMASS

## Installation des librairies Python

### Installation de l'installateur de librairies pip
Dans l'invité de commandes entrer : 
python get-pip.py

### Installation des librairies
pip install scipy
pip install matplotlib

### Installation de rasterio manuellement (visualisation des fichiers raster)

pip install rasterio-1.1.2-cp37-cp37m-win_amd64.whl

## Installation de GDAL pour Windows

### Etape 1 
https://sandbox.idre.ucla.edu/sandbox/tutorials/installing-gdal-for-windows

### Etape 2 
Dans l'invité de commandes entrer : 
setx GDAL_DATA "C:\Program Files\GDAL\gdal-data"
setx GDAL_DRIVER_PATH "C:\Program Files\GDAL\gdalplugins"
setx PROJ_LIB "C:\Program Files\GDAL\projlib"
setx PYTHONPATH "C:\Program Files\GDAL\"

### Visualiser des fichiers raster
https://automating-gis-processes.github.io/CSC18/lessons/L6/plotting-raster.html
