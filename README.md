### Übung 1 ###

* Punktewolke herunterladen z.B. aus dem [Kitty Datensatz](http://www.cvlibs.net/datasets/kitti/raw_data.php)
* Grundfläche wie z.B. Straße extrahieren

Grundfläche kann z.B. mit [RANSAC](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html) bestimmt werden.
Wichtige Hilfsfunktionen (z.B. Normalen bestimmen) sind im Repository pybind11_pcl (Python bindings für die PCL library) zu finden.