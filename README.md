# CESBIO_BIOMASS
Repo dédié au projet des méthodes de corrélation croisée adaptées à l'optimisation des corrections topographiques en imagerie radar à synthèse d'ouverture : applications aux images aéroportées bande P pour la mission BIOMASS

## Objectif du projet

Utiliser un facteur de correction angulaire pour corriger les erreurs des images radar. On procède par superposition de l'image du facteur de correction avec l'image radar.
Le but du projet et de proposer le meilleur moyen de superposer les deux images.

Les erreurs topographiques sont liées au choix de la fréquence du radar. Cette dernière étant trop grande, l'influence de la topographie est trop importante.
Le facteur de correction est calculé à l'aide des modèles numériques de surface (MNS) issues des missions SRTM et TDX)

Pour superposer les images on propose d'utiliser deux méthodes principales : 
- Normalized Cross Correlation (NCC)
- Recalage par algorithme de gradient 

## Ressources utiles

Article issue du sujet du projet sur les méthodes de comparaisons d'images : https://ieeexplore.ieee.org/document/4603110


## Installation des librairies Python

### Installation de l'installateur de librairies pip
Dans l'invité de commandes entrer : 
<pre><code>python get-pip.py
</code></pre>

### Installation des librairies
<pre><code>pip install scipy
pip install matplotlib
</code></pre>

### Installation de rasterio manuellement (visualisation des fichiers raster)
<pre><code>pip install rasterio-1.1.2-cp37-cp37m-win_amd64.whl
</code></pre>

## Installation de GDAL pour Windows

### Etape 1 
https://sandbox.idre.ucla.edu/sandbox/tutorials/installing-gdal-for-windows

### Etape 2 
Dans l'invité de commandes entrer : 
<pre><code>setx GDAL_DATA "C:\Program Files\GDAL\gdal-data"
setx GDAL_DRIVER_PATH "C:\Program Files\GDAL\gdalplugins"
setx PROJ_LIB "C:\Program Files\GDAL\projlib"
setx PYTHONPATH "C:\Program Files\GDAL\"
</code></pre>

### Visualiser des fichiers raster
https://automating-gis-processes.github.io/CSC18/lessons/L6/plotting-raster.html

### Corrélations croisées

https://askcodez.com/correlation-croisee-entre-deux-images.html


