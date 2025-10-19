"""Aufgabe: Öffnet pywebview mit myinstants.com.
Nutzer klickt selbst auf „Download“. Dein JS fängt nur den Klick ab, lädt MP3 in sounds/sound_cache/, fragt: „Zu welchem Key? (z. B. ok, thumbsup, laugh)“ und schreibt in sound_map.json und ins Profil.

Eingaben: MP3-URL vom Nutzer-Klick + frei gewählter Key.

Ausgaben: lokale MP3 + Mapping {key: relative_path}.

Wichtig: kein Auto-Massendownload; Download nur nach Nutzeraktion.
mehr recherche ob man überhaupt "einfach so" ein button einfügen darf etc. (rechtliches)."""

import webview
import requests
import os
import json
from utils.json_manager import save_json

# --- Ordnerstruktur anpassen ---
CACHE_DIR = os.path.join("sounds", "sound_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
SOUND_MAP_PATH = os.path.join("sounds", "sound_map.json")

# --------------------------------------------------
# 🧠 PyWebView API: verwaltet Download & Sound-Link
# --------------------------------------------------
class API:
    def __init__(self):
        self.last_url = None

    def set_sound(self, url):
        """Wird vom JS aufgerufen, wenn ein Sound abgespielt wird."""
        if url.startswith("/"):
            url = "https://www.myinstants.com" + url
        self.last_url = url
        print(f"🎧 SOUND DETECTED (NOT DOWNLOADED): {url}")
        return "SOUND DETECTED"

    def download_last(self):
        """Lädt den zuletzt erkannten Sound herunter und fragt nach einer Kategorie."""
        if not self.last_url:
            print("⚠️ No sound selected yet.")
            return "NO SOUND SELECTED"

        url = self.last_url
        filename = url.split("/")[-1]
        path = os.path.join(CACHE_DIR, filename)

        # Prüfen, ob bereits vorhanden
        if os.path.exists(path):
            print(f"ℹ️ {filename} already downloaded.")
            return "ALREADY DOWNLOADED"

        # Datei laden
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"✅ {filename} downloaded ({len(r.content)//1024} KB)")

            # --- Nutzer fragt, welchem Verhalten der Sound zugeordnet wird ---
            category = webview.windows[0].evaluate_js("prompt('🎵 For which behavior/emotion is this sound? (e.g. ok, laugh, angry)')")
            if not category:
                print("⚠️ No category entered, skipping.")
                return "SKIPPED"

            # Pfad speichern (relativ)
            rel_path = os.path.join("sounds", "sound_cache", filename)
            save_json(SOUND_MAP_PATH, category, rel_path)
            save_json("setup/setup_config.json", "sounds", {category: rel_path})

            print(f"💾 Sound '{filename}' saved under key '{category}'")
            return "OK"

        except Exception as e:
            print("❌ Error during download:", e)
            return str(e)


# --------------------------------------------------
# 🌐 Browser öffnen + UI-Button hinzufügen
# --------------------------------------------------
def run_sound_setup(user="default"):
    """Wird vom Setup-Wizard oder main.py aufgerufen."""
    api = API()

    window = webview.create_window(
        title="🎧 Sound Selection (Integrated)",
        url="https://www.myinstants.com/en/index/de/",
        width=1200,
        height=800,
        js_api=api
    )

    js_hook = r"""
        (function() {
            console.log('🚀 JS Hook active');
            const OriginalAudio = window.Audio;
            window.Audio = function(...args) {
                const audio = new OriginalAudio(...args);
                const origSrc = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, 'src');
                Object.defineProperty(audio, 'src', {
                    set(value) {
                        console.log('🎵 Sound detected:', value);
                        if (value && value.endsWith('.mp3')) {
                            try { window.pywebview.api.set_sound(value); }
                            catch (err) { console.error('Send to Python failed:', err); }
                        }
                        origSrc.set.call(this, value);
                    },
                    get() { return origSrc.get.call(this); }
                });
                return audio;
            };
            console.log('✅ Audio hook ready');
        })();
    """

    def inject_ui(window):
        window.evaluate_js(js_hook)
        # Floating Download-Button
        window.evaluate_js("""
            const btn = document.createElement('button');
            btn.innerText = '💾 Download last sound';
            btn.style.position = 'fixed';
            btn.style.top = '50%';
            btn.style.right = '25px';
            btn.style.transform = 'translateY(-50%)';
            btn.style.zIndex = '9999';
            btn.style.padding = '12px 20px';
            btn.style.background = '#00cc66';
            btn.style.border = 'none';
            btn.style.color = 'white';
            btn.style.fontSize = '16px';
            btn.style.borderRadius = '10px';
            btn.style.cursor = 'pointer';
            btn.onclick = () => window.pywebview.api.download_last();
            document.body.appendChild(btn);
        """)
        print("✅ UI Button added")

    webview.start(inject_ui, window)
    return True


if __name__ == "__main__":
    run_sound_setup()
