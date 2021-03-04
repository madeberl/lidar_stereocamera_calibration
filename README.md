## Objektdetektion in Punktwolken und Bildern ##
In der Bachelorarbeit soll ein Algorithmus entwickelt werden zur automatischen Detektion von Kugeln und Würfeln in Lidar Punktwolken und Kamerabildern. In einem weiteren Schritt sollen die Korrespondenzen zwischen den Modalitäten gefunden werden. Die korrespondierenden Punkte werden anschließend in Folgeprojekten zur Lidar-Kamera Kalibreirung genutzt.

### main.py ###
Bevor das Programm gestartet werden kann müssen ein paar Anpassungen
vorgenommen werden:
1. Datenpfade für Lidar und Stereo Daten eintragen. Es können mehrere Daten
gleichzeitig eingetragen werden. 
2. Die Werte für remove_points müssen angepasst werden, hier reicht meistens das löschen aller
Werte für y < 0. Eventuell ist aber etwas ausprobieren nötig. Falls auch Werte der x- oder z-Achse 
   mit in das Ausschneiden einbezogen werden sollen, sollte remove_points durch remove_points_extended
   ersetzt werden.

#### Options ####
````
-d --Debug: Debug on
-dl --distance_lidar:   Threshold for Distance Compute Function for Lidar
                        (Default: 0.05)
-ds --distance_stereo:  Threshold for Distance Compute Function for Stereo
                        (Default: 0.05)
-rl --ransac_lidar:     Threshold for Ransac - Lidar (Default: 0.01)
-rs --ransac_stereo:    Threshold for Ransac - Stereo (Default: 0.004)
````


### merge_files.py ###
Um mehrere .pcd und .txt Dateien miteinander zu verbinden, 
muss merge_files.py aufgerufen werden. Es wird ein Explorerfenster 
geöffnet, bei dem die zweitoberste Ebene ausgewählt werden muss
(wenn die pcd Daten z.B. in paket/seq1/lidar liegen, muss paket ausgewählt werden).
Das Programm durchsucht dann alle Ebenen und fasst die pcd Dateien 
zu einer merged.pcd zusammen und verfährt genauso mit den Textdateien.
Die Merge-Dateien liegen dann in den jeweiligen Ordnern.
