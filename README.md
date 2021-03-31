## Objektdetektion in Punktwolken und Bildern ##
In der Bachelorarbeit soll ein Algorithmus entwickelt werden zur automatischen Detektion von Kugeln und Würfeln in Lidar Punktwolken und Kamerabildern. In einem weiteren Schritt sollen die Korrespondenzen zwischen den Modalitäten gefunden werden. Die korrespondierenden Punkte werden anschließend in Folgeprojekten zur Lidar-Kamera Kalibreirung genutzt.

### main.py ###
Bevor das Programm gestartet werden kann müssen ein paar Anpassungen
vorgenommen werden:\
Es müssen die Datenpfade für Lidar und Stereo Daten als Liste im Format ````["data"] ````eintragen werden. 
Es können mehrere Daten gleichzeitig eingetragen werden, dann die Datensätze Komma-separiert eintragen. 
Zusätzlich ist es ratsam die Daten vorzubeschneiden, dafür werden die Argumente unten verwendet.
Im nachfolgenden Bild ist das verwendete Koordinatensystem zu sehen:\
![Koordinatensystem](https://git.ios.htwg-konstanz.de/mof-sprojekte/ba-matthias-deberling/-/tree/master/Bachelorarbeit/pictures/alpha_coordinate_2.png)
\
Zum Schluss sollten noch die Maße für das Paket angegeben werden, eine Übersicht ist im folgenden Bild zu sehen:\
![Paketmaße](https://git.ios.htwg-konstanz.de/mof-sprojekte/ba-matthias-deberling/-/tree/master/Bachelorarbeit/pictures/paket_2.png)
\
Wichtig ist, dass die Seiten fest sind. Wenn das Paket also anders als in der Grafik
aufgestellt wird, sollten die Maße so eingetragen werden.
#### Options ####
````
-d --Debug: Debug on
---------------------------------------------------------------------------
-dl --distance_lidar    Threshold for Distance Compute Function for Lidar
                        (Default: 0.1)
-ds --distance_stereo   Threshold for Distance Compute Function for Stereo
                        (Default: 0.05)
---------------------------------------------------------------------------
-rl --ransac_lidar      Threshold for Ransac - Lidar (Default: 0.01)
-rs --ransac_stereo     Threshold for Ransac - Stereo (Default: 0.004)
---------------------------------------------------------------------------
-xmin --x_minimum       Minimum value for cropping in x-direction
-xmax --x_maximum       Maximum value for cropping in x-direction
-ymin --y_minimum       Minimum value for cropping in y-direction
-ymax --y_maximum       Maximum value for cropping in y-direction
-zmin --z_minimum       Minimum value for cropping in z-direction
-zmax --z_maximum       Maximum value for cropping in z-direction
---------------------------------------------------------------------------
-l --length             Length of paket (Default: 0.45)
-w --width              Width of paket (Default: 0.35)
-he --height             Height of paket (Default: 0.4)
````


### merge_files.py ###
Um mehrere .pcd und .txt Dateien miteinander zu verbinden, 
muss merge_files.py aufgerufen werden. Es wird ein Datei-Explorerfenster 
geöffnet, bei dem die zweitoberste Ebene ausgewählt werden muss
(wenn die pcd Daten z.B. in paket/seq1/lidar liegen, muss paket ausgewählt werden).
Das Programm durchsucht dann alle Ebenen und fasst die pcd Dateien 
zu einer merged.pcd zusammen und verfährt genauso mit den Textdateien.
Die Merge-Dateien liegen dann in den jeweiligen Ordnern.
