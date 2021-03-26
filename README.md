## Objektdetektion in Punktwolken und Bildern ##
In der Bachelorarbeit soll ein Algorithmus entwickelt werden zur automatischen Detektion von Kugeln und Würfeln in Lidar Punktwolken und Kamerabildern. In einem weiteren Schritt sollen die Korrespondenzen zwischen den Modalitäten gefunden werden. Die korrespondierenden Punkte werden anschließend in Folgeprojekten zur Lidar-Kamera Kalibreirung genutzt.

### main.py ###
Bevor das Programm gestartet werden kann müssen ein paar Anpassungen
vorgenommen werden:\
Datenpfade für Lidar und Stereo Daten eintragen. Es können mehrere Daten
gleichzeitig eingetragen werden. Wenn mehrere Dateien eintragen werden müssen diese
mit , miteinander verbunden werden. Dann muss ebenfalls die Zeile 643 (Variable v)
ersetzt werden durch die Anzahl an Dateien + 1.


#### Options ####
````
-d --Debug: Debug on
-dl --distance_lidar:   Threshold for Distance Compute Function for Lidar
                        (Default: 0.1)
-ds --distance_stereo:  Threshold for Distance Compute Function for Stereo
                        (Default: 0.05)
-rl --ransac_lidar:     Threshold for Ransac - Lidar (Default: 0.01)
-rs --ransac_stereo:    Threshold for Ransac - Stereo (Default: 0.004)
-cr --crop:             Value for cropping data in y-axle
````


### merge_files.py ###
Um mehrere .pcd und .txt Dateien miteinander zu verbinden, 
muss merge_files.py aufgerufen werden. Es wird ein Datei-Explorerfenster 
geöffnet, bei dem die zweitoberste Ebene ausgewählt werden muss
(wenn die pcd Daten z.B. in paket/seq1/lidar liegen, muss paket ausgewählt werden).
Das Programm durchsucht dann alle Ebenen und fasst die pcd Dateien 
zu einer merged.pcd zusammen und verfährt genauso mit den Textdateien.
Die Merge-Dateien liegen dann in den jeweiligen Ordnern.
