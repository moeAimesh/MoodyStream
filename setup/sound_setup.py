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
from utils.json_manager import save_json, update_json
from utils.settings import CACHE as CACHE_DIR, SOUND_MAP_PATH, SETUP_CONFIG_PATH, ALLOWED_BEHAVIOUR_KEYS, MYINSTANTS_URL


# --- Ordnerstruktur anpassen ---
os.makedirs(CACHE_DIR, exist_ok=True)


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
        """Lädt den zuletzt erkannten Sound herunter (falls nötig) und fragt nach einer Kategorie.
           Speichert Mapping in sounds/sound_map.json UND setup/setup_config.json (merge)."""
        if not self.last_url:
            print("⚠️ No sound selected yet.")
            return "NO SOUND SELECTED"

        url = self.last_url
        # robuste Dateinamenermittlung
        name = url.split("?")[0].split("/")[-1] or "sound.mp3"
        if not name.lower().endswith(".mp3"):
            name += ".mp3"
        path = os.path.join(CACHE_DIR, name)

        # Falls Datei schon existiert, nicht abbrechen – weiter mappen
        if not os.path.exists(path):
            try:
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                with open(path, "wb") as f:
                    f.write(r.content)
                print(f"✅ {name} downloaded ({len(r.content)//1024} KB)")
            except Exception as e:
                print(" Error during download:", e)
                return str(e)
        else:
            print(f"ℹ️ {name} already exists, will only map to a key.")

        # --- Nutzer fragt, welchem Verhalten der Sound zugeordnet wird ---
        placeholder = "happy"
        allowed_str = ", ".join(ALLOWED_BEHAVIOUR_KEYS) if ALLOWED_BEHAVIOUR_KEYS else "any"
        prompt = f"🎵 For which behavior/emotion is this sound?\nAllowed: {allowed_str}"
        category = webview.windows[0].evaluate_js(f"prompt({json.dumps(prompt)}, {json.dumps(placeholder)})")

        if not category:
            print("⚠️ No category entered, skipping.")
            return "SKIPPED"

        category = str(category).strip().lower()
        if ALLOWED_BEHAVIOUR_KEYS and category not in ALLOWED_BEHAVIOUR_KEYS:
            msg = f"Invalid key: {category}"
            print("x", msg)
            return msg

        # Pfad relativ (forward slashes)
        rel_path = "sounds/sound_cache/" + os.path.basename(path)

        # 1) In sound_map.json -> einzelner Key
        save_json(str(SOUND_MAP_PATH), category, rel_path)

        # 2) In setup_config.json -> Mergen unter "sounds"
        update_json(str(SETUP_CONFIG_PATH), "sounds", {category: rel_path})

        print(f"💾 Sound '{name}' saved under key '{category}'")
        return "OK"



# --------------------------------------------------
# 🌐 Browser öffnen + UI-Button hinzufügen
# --------------------------------------------------
def run_sound_setup(user="default"):
    """Wird vom Setup-Wizard oder main.py aufgerufen."""
    api = API()

    window = webview.create_window(
        title="🎧 Sound Selection (Integrated)",
        url=MYINSTANTS_URL,
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
