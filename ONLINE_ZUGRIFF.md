# Online-Zugriff einrichten - Kurzanleitung

## 🚀 In 5 Minuten online!

### Schritt 1: GitHub Account erstellen (falls noch nicht vorhanden)
1. Gehe zu [github.com/signup](https://github.com/signup)
2. Erstelle einen kostenlosen Account

### Schritt 2: Repository erstellen
1. Klicke auf [github.com/new](https://github.com/new)
2. Repository Name: `osp-anlaufanalyse`
3. Wähle **Public** (für kostenloses Hosting)
4. Klicke "Create repository"

### Schritt 3: Code hochladen

**Einfachste Methode - GitHub Web Interface:**

1. Auf der Repository-Seite: Klicke "uploading an existing file"
2. Ziehe ALLE Dateien aus dem `OSP_New` Ordner in den Browser
   - **WICHTIG:** NICHT den `venv/` Ordner hochladen!
   - Lade hoch: `streamlit_dashboard.py`, `analyze_movement_data.py`, `requirements.txt`, `Input files/`, etc.
3. Commit message: "Initial commit"
4. Klicke "Commit changes"

### Schritt 4: Streamlit Cloud Deployment

1. Gehe zu [share.streamlit.io](https://share.streamlit.io)
2. Klicke "Sign in with GitHub"
3. Klicke "New app"
4. Einstellungen:
   - **Repository:** `DEIN-USERNAME/osp-anlaufanalyse`
   - **Branch:** `main`
   - **Main file path:** `streamlit_dashboard.py`
5. Klicke "Deploy!"

### Schritt 5: Fertig! 🎉

Nach 2-3 Minuten ist deine App online unter:
```
https://DEIN-USERNAME-osp-anlaufanalyse.streamlit.app
```

**Diesen Link kannst du an deinen Kunden schicken!**

---

## 📱 Lokales Testen (vor dem Hochladen)

```bash
cd OSP_New
source venv/bin/activate  # Mac/Linux
# oder: venv\Scripts\activate  # Windows

streamlit run streamlit_dashboard.py
```

Die App öffnet sich automatisch im Browser unter `http://localhost:8501`

---

## ⚙️ Was wurde geändert?

### Neue Dateien:
- ✅ `streamlit_dashboard.py` - Online-fähige Version des Dashboards
- ✅ `.streamlit/config.toml` - Streamlit-Konfiguration (OSP-Farben)
- ✅ `.gitignore` - Verhindert Upload unnötiger Dateien
- ✅ `DEPLOYMENT.md` - Ausführliche Anleitung
- ✅ `ONLINE_ZUGRIFF.md` - Diese Kurzanleitung

### Aktualisierte Dateien:
- ✅ `requirements.txt` - Streamlit hinzugefügt

### Alte Dateien (bleiben erhalten):
- ✅ `analyze_movement_data.py` - Unverändert
- ✅ `movement_analysis_dashboard.py` - Dash-Version (lokal)
- ✅ `movement_analysis_dashboard_v2.py` - Dash-Version v2 (lokal)

---

## 🎨 Features der Streamlit-Version

### Layout (wie vom Kunden gewünscht):
- **Links (30%):** Liste aller Versuche
  - Spalten: Athlet | Versuch | Lücken | Qualität
  - Farbcodierung: Grün/Gelb/Rot
  - Filter nach Qualität & Disziplin
  - Sortierbar

- **Rechts (70%):** Detailanalyse
  - Große interaktive Plots (Plotly)
  - Status-Badge (🟢/🟡/🔴)
  - 3 Spalten mit Metriken:
    - Qualitätsmetriken
    - Zone 11-6m
    - Zone 6-1m
  - Lücken-Tabelle

### Optimierungen:
- ✅ Optimiert für FHD (1920x1080)
- ✅ Responsive Design
- ✅ OSP-Branding (Logo, Farben)
- ✅ Caching für schnelle Ladezeiten
- ✅ Geschwindigkeitsfilterung dokumentiert

---

## 🔒 Datenschutz

**Wichtig:** Streamlit Cloud Public Apps sind öffentlich zugänglich!

### Option 1: Öffentlich (kostenlos)
- Jeder mit dem Link kann die App nutzen
- Gut für: Demos, unkritische Daten

### Option 2: Mit Passwortschutz (kostenlos)
Füge am Anfang von `streamlit_dashboard.py` ein:

```python
import streamlit as st

def check_password():
    def password_entered():
        if st.session_state["password"] == "OSP2025":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Passwort", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Passwort", type="password", on_change=password_entered, key="password")
        st.error("Falsches Passwort")
        return False
    else:
        return True

if not check_password():
    st.stop()

# Rest der App...
```

### Option 3: Privat (kostenpflichtig)
- Streamlit Cloud: $20/Monat für private Apps
- Nur eingeladene Nutzer haben Zugriff

---

## 🆘 Häufige Probleme

### "ModuleNotFoundError: No module named 'analyze_movement_data'"
**Lösung:** Stelle sicher, dass `analyze_movement_data.py` im gleichen Ordner wie `streamlit_dashboard.py` liegt.

### "FileNotFoundError: Input files"
**Lösung:** Der `Input files/` Ordner muss mit hochgeladen werden!

### App lädt sehr langsam
**Lösung:** 
- Erste Ladung dauert länger (Caching wird aufgebaut)
- Streamlit Cloud Free Tier hat begrenzte Ressourcen
- Nach Inaktivität "schläft" die App (normal im Free Tier)

### "This app has exceeded its resource limits"
**Lösung:** 
- Zu viele/große Dateien
- Erwäge, nur eine Auswahl hochzuladen
- Oder upgrade auf Streamlit Cloud Plus ($20/Monat)

---

## 📞 Support

**Bei Problemen:**
1. Prüfe die [Streamlit Docs](https://docs.streamlit.io)
2. Schau in die ausführliche `DEPLOYMENT.md`
3. Streamlit Forum: [discuss.streamlit.io](https://discuss.streamlit.io)

---

## 🔄 Updates durchführen

**Nach Code-Änderungen:**
1. Lade die geänderten Dateien auf GitHub hoch
2. Streamlit Cloud deployed automatisch neu (dauert 2-3 Min)
3. Fertig!

**Oder über Kommandozeile:**
```bash
git add .
git commit -m "Update XYZ"
git push
```

---

## ✅ Checkliste vor dem Hochladen

- [ ] `venv/` Ordner NICHT hochladen
- [ ] `Input files/` Ordner MIT hochladen
- [ ] `requirements.txt` vorhanden
- [ ] `streamlit_dashboard.py` vorhanden
- [ ] `analyze_movement_data.py` vorhanden
- [ ] `.gitignore` vorhanden
- [ ] Lokal getestet (`streamlit run streamlit_dashboard.py`)

---

**Viel Erfolg! Bei Fragen einfach melden. 🚀**


