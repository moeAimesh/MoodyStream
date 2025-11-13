from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import requests
import webview
from utils.json_manager import save_json, update_json
from utils.settings import (
    ALLOWED_BEHAVIOUR_KEYS,
    MYINSTANTS_URL,
    SETUP_CONFIG_PATH,
    SOUND_CACHE_DIR,
    SOUND_MAP_PATH,
)
CACHE_DIR = Path(SOUND_CACHE_DIR)
SOUND_MAP_FILE = Path(SOUND_MAP_PATH)
SETUP_CONFIG_FILE = Path(SETUP_CONFIG_PATH)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
def _normalize_url(url: str) -> str:
    if url.startswith("/"):
        return "https://www.myinstants.com" + url
    return url
class API:
    """JS bridge: receives detected mp3 URLs and triggers downloads."""
    def __init__(self) -> None:
        self.last_url: Optional[str] = None
    def set_sound(self, url: str) -> str:
        """Called from JS whenever an mp3 URL is detected."""
        try:
            url = _normalize_url(url)
            if not url.lower().endswith(".mp3"):
                print(f"[sound_setup] ignored non-mp3 URL: {url}")
                return "IGNORED"
            self.last_url = url
            print(f"[sound_setup] detected mp3: {url}")
            return "SOUND DETECTED"
        except Exception as exc:  # pragma: no cover - defensive
            print("[sound_setup] set_sound error:", exc)
            return "ERROR"
    def download_last(self) -> str:
        """
        Download the last detected sound if necessary and store its mapping.
        """
        if not self.last_url:
            print("[sound_setup] No sound selected yet.")
            return "NO SOUND SELECTED"
        url = self.last_url
        name = url.split("?")[0].split("/")[-1] or "sound.mp3"
        if not name.lower().endswith(".mp3"):
            name += ".mp3"
        target = CACHE_DIR / name
        if not target.exists():
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                target.write_bytes(response.content)
                print(f"[sound_setup] downloaded {name} ({len(response.content)//1024} KB)")
            except Exception as exc:
                print("[sound_setup] error during download:", exc)
                return str(exc)
        else:
            print(f"[sound_setup] {name} already exists locally; reusing file.")
        allowed_str = ", ".join(ALLOWED_BEHAVIOUR_KEYS) if ALLOWED_BEHAVIOUR_KEYS else "any"
        prompt = (
            "For which behavior/emotion is this sound?\n"
            f"Allowed: {allowed_str}"
        )
        placeholder = "happy"
        category = webview.windows[0].evaluate_js(
            f"prompt({json.dumps(prompt)}, {json.dumps(placeholder)})"
        )
        if not category:
            print("[sound_setup] No category entered, skipping.")
            return "SKIPPED"
        category = str(category).strip().lower()
        if ALLOWED_BEHAVIOUR_KEYS and category not in ALLOWED_BEHAVIOUR_KEYS:
            msg = f"Invalid key: {category}"
            print("[sound_setup]", msg)
            return msg
        rel_path = "sounds/sound_cache/" + target.name
        save_json(str(SOUND_MAP_FILE), category, rel_path)
        update_json(str(SETUP_CONFIG_FILE), "sounds", {category: rel_path})
        print(f"[sound_setup] mapped '{name}' to '{category}'")
        return "OK"
JS_HOOK = r"""
(function () {
    console.log('[Moody] JS hook active');
    function absoluteUrl(u) {
        try { return new URL(u, location.origin).toString(); }
        catch (e) { return u; }
    }
    function report(u) {
        if (!u) return;
        const url = absoluteUrl(u);
        if (!/\.mp3(\?|$)/i.test(url)) return;
        console.log('[Moody] mp3 detected:', url);
        if (window.pywebview && window.pywebview.api && window.pywebview.api.set_sound) {
            window.pywebview.api.set_sound(url);
        }
    }
    const originalFetch = window.fetch;
    window.fetch = async function(resource, init) {
        try {
            if (typeof resource === 'string') { report(resource); }
            else if (resource && resource.url) { report(resource.url); }
        } catch (e) {}
        return originalFetch.apply(this, arguments);
    };
    const originalOpen = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function(method, url) {
        try { report(url); } catch (e) {}
        return originalOpen.apply(this, arguments);
    };
    const targets = [HTMLMediaElement.prototype];
    [HTMLAudioElement, HTMLVideoElement].forEach(proto => {
        if (proto && proto.prototype) { targets.push(proto.prototype); }
    });
    targets.forEach(proto => {
        const desc = Object.getOwnPropertyDescriptor(proto, 'src');
        if (desc && desc.set) {
            Object.defineProperty(proto, 'src', {
                set(value) { try { report(value); } catch (e) {} return desc.set.call(this, value); },
                get() { return desc.get.call(this); }
            });
        }
    });
    const observer = new MutationObserver(mutations => {
        mutations.forEach(mutation => {
            mutation.addedNodes.forEach(node => {
                if (!node.tagName) { return; }
                const tag = node.tagName.toLowerCase();
                if ((tag === 'audio' || tag === 'video' || tag === 'source') && node.src) {
                    report(node.src);
                }
                if (node.querySelectorAll) {
                    node.querySelectorAll('a[href]').forEach(a => report(a.href));
                    node.querySelectorAll('source[src]').forEach(s => report(s.src));
                }
            });
        });
    });
    observer.observe(document.documentElement || document.body, { childList: true, subtree: true });
    document.querySelectorAll('a[href], source[src], audio[src], video[src]').forEach(el => {
        if (el.href) { report(el.href); }
        if (el.src) { report(el.src); }
    });
    console.log('[Moody] JS hook installed');
})();
"""
def run_sound_setup(user: str = "default") -> bool:
    """Entry point used by the setup wizard."""
    api = API()
    window = webview.create_window(
        title="[Moody] Sound Selection",
        url=MYINSTANTS_URL,
        width=1200,
        height=800,
        js_api=api,
    )
    def inject_ui(win: webview.Window) -> None:
        win.evaluate_js(JS_HOOK)
        win.evaluate_js(
            """
            (function () {
                const btn = document.createElement('button');
                btn.textContent = 'Download last sound';
                btn.style.position = 'fixed';
                btn.style.top = '50%';
                btn.style.right = '25px';
                btn.style.transform = 'translateY(-50%)';
                btn.style.zIndex = 9999;
                btn.style.padding = '12px 20px';
                btn.style.background = '#00cc66';
                btn.style.border = 'none';
                btn.style.color = '#fff';
                btn.style.fontSize = '16px';
                btn.style.borderRadius = '10px';
                btn.style.cursor = 'pointer';
                btn.onclick = () => window.pywebview.api.download_last();
                document.body.appendChild(btn);
            })();
            """
        )
    webview.start(inject_ui, window)
    return True
if __name__ == "__main__":
    run_sound_setup()