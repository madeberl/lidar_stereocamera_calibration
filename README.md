## Pylonendetektion in Punktwolken und Bildern ##
In der Bachelorarbeit soll ein Algorithmus entwickelt werden zur automatischen Detektion von Straßenpylonen in Lidar Punktwolken und Kamerabildern. In einem weiteren Schritt sollen die Korrespondenzen zwischen den Modalitäten gefunden werden.
![alt text](https://unitbase.de/image/cache/catalog/Verkehr/pylone-leitkegel-mieten-berlin-unitbase-800x800.jpg)

### Übung 1 ###

* Punktewolke herunterladen z.B. aus dem [Kitty Datensatz](http://www.cvlibs.net/datasets/kitti/raw_data.php)
* Punktwolke mit numpy einlesen
* Punktwolke visualisieren (matplotlib, cloud compare, )
* Grundfläche wie z.B. Straße extrahieren

Grundfläche kann z.B. mit [RANSAC](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html) bestimmt werden.
Wichtige Hilfsfunktionen (z.B. Normalen bestimmen) sind im Repository pybind11_pcl (Python bindings für die PCL library) zu finden.