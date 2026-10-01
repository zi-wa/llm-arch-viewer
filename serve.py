"""Local server for LLM Architecture Viewer. Standard library only.

Serves the folder on 127.0.0.1 and proxies /hf/... to huggingface.co with this PC's Hugging Face token
(HF_TOKEN, else HF_HOME/token, else ~/.cache/huggingface/token), so gated repos whose terms the account
accepted load their own config.json. The token stays in this process; the page never sees it.
"""
import http.server
import os
import re
import socket
import sys
import urllib.error
import urllib.request
import webbrowser

Root = os.path.dirname(os.path.abspath(__file__))
PreferredPort = 47652
PortAttempts = 20
HfBase = "https://huggingface.co"
# Only what the page reads: model search, model metadata and a repo's config.json.
ProxyPath = re.compile(r"^/hf/(api/models|api/models/[\w.-]+/[\w.-]+|[\w.-]+/[\w.-]+/resolve/main/config\.json)$")
TimeoutSec = 30

MimeTypes = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".md": "text/plain; charset=utf-8",
}


def read_token():
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    hf_home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    token_file = os.path.join(hf_home, "token")
    if os.path.isfile(token_file):
        with open(token_file, encoding="utf-8") as fh:
            return fh.read().strip() or None
    return None


def pick_port():
    for port in range(PreferredPort, PreferredPort + PortAttempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port between %d and %d" % (PreferredPort, PreferredPort + PortAttempts - 1))


def make_handler(token, port):
    allowed_hosts = {"127.0.0.1:%d" % port, "localhost:%d" % port}

    class Handler(http.server.SimpleHTTPRequestHandler):
        extensions_map = MimeTypes

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=Root, **kwargs)

        def end_headers(self):
            self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def log_message(self, format, *args):
            pass

        def do_GET(self):
            path, _, query = self.path.partition("?")
            if not path.startswith("/hf/"):
                super().do_GET()
                return
            # A rebinding page on another origin would arrive with its own Host; the token only serves this one.
            if self.headers.get("Host") not in allowed_hosts or not ProxyPath.match(path):
                self.send_error(403)
                return
            url = HfBase + path[3:] + ("?" + query if query else "")
            headers = {"User-Agent": "LlmArchViewer/1"}
            if token:
                headers["Authorization"] = "Bearer " + token
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=TimeoutSec) as res:
                    status, body = res.status, res.read()
            except urllib.error.HTTPError as err:
                status, body = err.code, err.read()
            except urllib.error.URLError as err:
                status, body = 502, str(err.reason).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    token = read_token()
    port = pick_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), make_handler(token, port))
    url = "http://127.0.0.1:%d/" % port
    print("serving at %s (Hugging Face token: %s)" % (url, "found" if token else "none, public repos only"), flush=True)
    if "--serve-only" not in sys.argv:
        webbrowser.open(url)
        print("close this window to stop the server", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
