# Anlaufanalyse-Tool

Dieses Tool analysiert Anlaufgeschwindigkeitsdaten der Deutschen Jugendmeisterschaften im Weit- und Dreisprung. Es erkennt Lücken in den Messdaten, berechnet Geschwindigkeiten und Schrittmuster und erstellt Visualisierungen.

## Voraussetzungen
- Python 3.8 oder neuer
- pip (Python Package Manager)

## Installation

### 1. Projektordner herunterladen
Lade den gesamten Projektordner `OSP_New` auf deinen Rechner und entpacke ihn, falls nötig.

### 2. Abhängigkeiten installieren (WICHTIG!)
Öffne die Kommandozeile (Terminal) und wechsle in den Projektordner:

**Windows:**
```bash
cd C:\Pfad\zu\OSP_New
```

**Mac/Linux:**
```bash
cd /Pfad/zu/OSP_New
```

**Virtuelle Umgebung erstellen (empfohlen):**
```bash
python3 -m venv venv
```

**Virtuelle Umgebung aktivieren:**
- **Mac/Linux:** `source venv/bin/activate`
- **Windows:** `venv\Scripts\activate`

**Pakete installieren:**
```bash
pip install -r requirements.txt
```

**Hinweis:** Dieser Schritt ist zwingend erforderlich! Ohne die Installation der Pakete (numpy, pandas, matplotlib, plotly, dash, pyts, etc.) funktioniert das Tool nicht.

**Bei Problemen mit "externally-managed-environment":**
Nutze die virtuelle Umgebung wie oben beschrieben. Dies ist die empfohlene Methode und vermeidet Konflikte mit System-Python.

## Nutzung

### Hauptanalyse-Tool
Das Hauptskript analysiert alle `.dat`-Dateien in den Input-Ordnern und erstellt:
- PNG-Visualisierungen für jede Datei
- CSV-Report mit allen Lücken und Status-Informationen
- Detaillierte Konsolenausgabe

**Starten:**
```bash
source venv/bin/activate  # Falls virtuelle Umgebung genutzt wird
python analyze_movement_data.py
```

Die Analyse läuft über alle Ordner:
- `Input files/Drei M/` (Dreisprung Männer)
- `Input files/Drei W/` (Dreisprung Frauen)
- `Input files/Weit M/` (Weitsprung Männer)
- `Input files/Weit W/` (Weitsprung Frauen)

**Ergebnisse:**
- PNG-Dateien: `Input files/<Ordner>/<Dateiname>_analysis.png`
- CSV-Report: `gap_status_report.csv`

### Interaktives Dashboard
Für eine interaktive Analyse mit Web-Interface:

**Starten:**
```bash
source venv/bin/activate  # Falls virtuelle Umgebung genutzt wird
python movement_analysis_dashboard.py
```

Das Dashboard öffnet sich automatisch im Browser unter `http://localhost:8050`

**Features:**
- Einzelanalyse mit interaktiven Plots
- Übersicht aller Versuche
- Zonenanalyse (11-6m, 6-1m)
- Qualitätsmetriken
- Export-Funktionen (PDF, Excel, Präsentation)

## Datenstruktur

Die `.dat`-Dateien haben folgendes Format:
- Zeile 1: Datum und Uhrzeit
- Zeile 2: Athlet/Kommentar
- Zeile 3: Versuchsnummer
- Zeile 4: Absprungpunkt in Metern (z.B. `49.680` = 49680 mm)
- Zeile 5-8: Leer/Metadaten
- Ab Zeile 9: Messwerte in mm

## Hinweise
- Das Tool benötigt keine Internetverbindung und läuft komplett lokal
- Die Analyse kann je nach Anzahl der Dateien einige Minuten dauern
- Bei Fragen oder Problemen bitte melden

## Entwickelt für
OSP Hessen - Deutsche Jugendmeisterschaften 2025 