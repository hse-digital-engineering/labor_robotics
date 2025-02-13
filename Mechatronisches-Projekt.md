
# Mechatronisches Projekt: Autonome Fernsteuerung eines Unitree Go2 mit Kollisionsvermeidung

## Projektbeschreibung und Zielsetzung

Das Ziel dieses Projekts ist die Entwicklung einer Fernsteuerungsanwendung für den quadrupedalen Roboter **Unitree Go2**, mit der der Roboter über eine **Tastatur oder einen Game-Controller** navigiert werden kann. Gleichzeitig wird der **Kamerastream** des Roboters in Echtzeit auf einem Monitor angezeigt. Eine Kernfunktion des Projekts ist die **Kollisionsvermeidung** basierend auf den **Laserscanner-Daten**, um sicherzustellen, dass der Roboter einen definierten Sicherheitsabstand zu Hindernissen nicht unterschreitet. Zusätzlich soll der Roboter seine eigene **Geometrie ins Kamerabild projizieren**, um dem Bediener visuelles Feedback über die Passierbarkeit enger Durchgänge zu geben.

Das Projekt soll den Studenten eine praxisnahe Einführung in die Themen **Robotik, Sensordatenverarbeitung und Computer-Vision** geben und gleichzeitig Fähigkeiten in **Softwareentwicklung** und **mechatronischer Systemintegration** fördern.

![](Go2-Air.png)

## Verwendete Hardware
- **Unitree Go2** - Quadrupedaler Roboter
  - **LiDAR-Scanner** für Umgebungswahrnehmung
  - **RGB-Kamera** zur Live-Übertragung des Videostreams
- **Game-Controller oder Tastatur** für die Fernsteuerung
- **NVidia Jetson Nano Orin**  Entwicklungsplattform (alternativ eigener Laptop)

## Verwendete Software
- **ROS 2** 
- **WebRTC** 
- **Python** (optional C++ oder in Kombination)
- **REST API / ROS 2 Bridge** 
- **OpenCV**
- **Docker**
- **Linux (Ubuntu)**

## Aufgabenpakete
1. **Einrichtung der Fernsteuerung**
   - Implementierung der Steuerung über Tastatur und Game-Controller
   - Übersetzung der Eingaben in Bewegungsbefehle für den Roboter

2. **Kamerakalibrierung**
   - Aufnahme von Kalibrierbildern mit Schachbrettmuster oder Charuco-Board
   - Bestimmung der Kameramatrix und Verzerrungsparameter mit OpenCV
   - Anwendung der Kalibrierung auf den Live-Videostream zur Korrektur von Verzerrungen
   
3. **Live-Kamerastream mit WebRTC**
   - Zugriff auf den RGB-Kamerastream
   - Echtzeitübertragung des Videostreams an einen Monitor
   
4. **Kollisionsvermeidung mit LiDAR**
   - Verarbeitung der Laserscanner-Daten
   - Definition eines Sicherheitsabstands und Notfallstopps
   - Visuelle oder andere Warnungen bei zu nahen Hindernissen
   
5. **Geometrieprojektion in das Kamerabild**
   - Berechnung der Robotergeometrie und Anzeige im Livebild
   - Projektion von zwei aufeinander zulaufenden Linien, die die Breite des Roboters symbolisieren

6. **Visualisierung der Sensor-Daten mit RViz**
   - Visualisieren der Laser-Scanner-Daten relativ zum Roboter-Modell
   
6. **Integration und Tests**
   - Zusammenspiel aller Module testen und debuggen
   - Durchführung von Experimenten in realen Szenarien


## Literatur:  

Kamera: 
- [https://de.mathworks.com/help/vision/ug/camera-calibration.html](https://de.mathworks.com/help/vision/ug/camera-calibration.html)
- [https://wiki.ros.org/camera_calibration](https://wiki.ros.org/camera_calibration)

ROS:
- https://docs.ros.org/en/iron/index.html

Go2:
- https://github.com/legion1581/go2_webrtc_connect
- https://github.com/abizovnuralem/go2_ros2_sdk/tree/chore/update-docker-to-ros-humble




