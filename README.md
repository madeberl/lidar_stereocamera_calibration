## Pylonendetektion in Punktwolken und Bildern ##
In der Bachelorarbeit soll ein Algorithmus entwickelt werden zur automatischen Detektion von Straßenpylonen in Lidar Punktwolken und Kamerabildern. In einem weiteren Schritt sollen die Korrespondenzen zwischen den Modalitäten gefunden werden. Die korrespondierenden Punkte werden anschließend in Folgeprojekten zur Lidar-Kamera Kalibreirung genutzt.
![alt text](https://unitbase.de/image/cache/catalog/Verkehr/pylone-leitkegel-mieten-berlin-unitbase-800x800.jpg)

### Übung 1 ###

* Punktewolke herunterladen z.B. aus dem [Kitty Datensatz](http://www.cvlibs.net/datasets/kitti/raw_data.php)
* Punktwolke mit numpy einlesen
* Punktwolke visualisieren (matplotlib, cloud compare, meshlab, open3d)
* Hintergrund bestimmen, um freistehende Objekte wie Kegel zu extrahieren (z.B. mit [Open3d](http://www.open3d.org/docs/release/index.html) (compute_point_cloud_distance)). Beispiel [Personendetektion](https://www.blickfeld.com/de/blog/objektdetektion/)

Wichtige Hilfsfunktionen (z.B. Normalen bestimmen) sind im Repository pybind11_pcl (Python bindings für die PCL library) zu finden.

### Thesis ###
1. Datensatz aufnehmen (z.B. Pilonen im Innenhof der HTWG aufstellen)