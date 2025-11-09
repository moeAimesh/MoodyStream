"""Aufgabe: Öffnet PyWebView mit myinstants.com. Ein JS-Hook erkennt alle MP3-Aufrufe
(fetch, XHR, src-Zuweisungen, dynamische <source>-Einfügungen) und meldet die URL an Python.
Über den Button „Download last sound“ wird die zuletzt erkannte MP3 (falls nötig) geladen,
der Nutzer gibt einen Key (z. B. ok, thumbsup, happy) ein und die Zuordnung wird gespeichert.

Eingaben: automatisch erkannte MP3-URL (durch Nutzeraktion auf myinstants) + frei gewählter Key.

Ausgaben: lokale MP3 in sounds/sound_cache/ + Mapping {key: relative_path} in
- sounds/sound_map.json (einzelner Key)
- setup/setup_config.json unter "sounds" (Merge, keine Überschreibung)

Verhalten:
- Nur Nutzeraktionen: kein Auto-Massendownload.
- Bereits vorhandene Dateien werden nicht erneut geladen, können aber neu gemappt werden.
- URLs werden normalisiert (relative → absolute), nur .mp3 wird akzeptiert.

Hinweis (rechtlich/technisch):
- Nur einzelne, manuell ausgelöste Downloads verwenden; keine automatisierte Sammlung.
- Prüfe Nutzungsrechte/Lizenzen der Sounds bei externer/öffentlicher Verwendung.
"""

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
        """Wird vom JS aufgerufen, sobald irgendwo eine MP3-URL erkannt wird."""
        try:
            # relative → absolute URL auflösen
            if url.startswith("/"):
                url = "https://www.myinstants.com" + url
            # Safety: nur mp3 akzeptieren
            if not url.lower().endswith(".mp3"):
                print(f"⚠️ Ignored non-mp3 url: {url}")
                return "IGNORED"
            self.last_url = url
            print(f"🎧 SOUND DETECTED (NOT DOWNLOADED): {url}")
            return "SOUND DETECTED"
        except Exception as e:
            print("❌ set_sound error:", e)
            return "ERROR"

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
            (function(){
            console.log('🚀 Network+DOM Hook loaded');

            // ---- Helpers ----
            function absUrl(u){
                try { return new URL(u, location.origin).toString(); }
                catch(e){ return u; }
            }
            function report(u){
                try{
                if(!u) return;
                const url = absUrl(u);
                if(!/\.mp3(\?|$)/i.test(url)) return;
                console.log('🎵 MP3 detected:', url);
                if (window.pywebview && window.pywebview.api && window.pywebview.api.set_sound){
                    window.pywebview.api.set_sound(url);
                    toast('🎧 erkannt: '+url.split('/').pop());
                }
                }catch(e){ console.error(e); }
            }
            function toast(msg){
                const div=document.createElement('div');
                div.textContent=msg;
                Object.assign(div.style,{
                position:'fixed',bottom:'12px',left:'12px',background:'rgba(0,0,0,0.8)',
                color:'#fff',padding:'6px 10px',borderRadius:'6px',zIndex:999999,
                font:'12px/1.3 system-ui, sans-serif'
                });
                document.body.appendChild(div);
                setTimeout(()=>div.remove(),2200);
            }

            // ---- 1) fetch() hooken ----
            const _fetch = window.fetch;
            window.fetch = async function(resource, init){
                try{
                if (typeof resource === 'string') report(resource);
                else if (resource && resource.url) report(resource.url);
                }catch(e){}
                return _fetch.apply(this, arguments);
            };

            // ---- 2) XHR hooken ----
            const _open = XMLHttpRequest.prototype.open;
            XMLHttpRequest.prototype.open = function(method, url){
                try{ report(url); }catch(e){}
                return _open.apply(this, arguments);
            };

            // ---- 3) src-Zuweisungen (Audio/Video/Source) beobachten ----
            const targets = [HTMLMediaElement.prototype, HTMLAudioElement?.prototype, HTMLVideoElement?.prototype].filter(Boolean);
            targets.forEach(proto=>{
                const desc = Object.getOwnPropertyDescriptor(proto, 'src');
                if(desc && desc.set){
                Object.defineProperty(proto, 'src', {
                    set(v){ try{ report(v); }catch(e){}; return desc.set.call(this, v); },
                    get(){ return desc.get.call(this); }
                });
                }
            });

            // ---- 4) MutationObserver: spätes Einfügen von <source src="...mp3"> ----
            const mo = new MutationObserver(muts=>{
                for(const m of muts){
                for(const n of m.addedNodes){
                    if(n && n.tagName){
                    if(n.tagName.toLowerCase()==='source' && n.src) report(n.src);
                    if((n.tagName.toLowerCase()==='audio' || n.tagName.toLowerCase()==='video') && n.src) report(n.src);
                    // Links, die evtl. erst später gerendert werden
                    if(n.querySelectorAll){
                        n.querySelectorAll('a[href]').forEach(a=>{
                        if(/\/media\/sounds\/.+\.mp3(\?|$)/i.test(a.href) || /\.mp3(\?|$)/i.test(a.href)) report(a.href);
                        });
                        n.querySelectorAll('source[src]').forEach(s=>report(s.src));
                    }
                    }
                }
                }
            });
            mo.observe(document.documentElement || document.body, { childList:true, subtree:true });

            // ---- 5) Initiale DOM-Abtastung ----
            document.querySelectorAll('a[href], source[src], audio[src], video[src]').forEach(el=>{
                if (el.href) report(el.href);
                if (el.src)  report(el.src);
            });

            console.log('✅ Network+DOM Hook installed');
            })();
            """


    print("🔍 Injecting JS into myinstants...")
    
    def inject_ui(window):
        window.evaluate_js(js_hook)
        print("✅ JS Hook injected")
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
