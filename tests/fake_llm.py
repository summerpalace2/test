"""本地假大模型服务（OpenAI 兼容 /chat/completions，SSE 流式），记录收到的请求用于断言。"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class FakeLLM:
    def __init__(self, reply="这是模型的回答：V8 的重要性最高，AUC 约 0.97。", fail_status=None):
        self.requests: list[dict] = []
        self.reply = reply
        self.fail_status = fail_status
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):  # GET /models：模型列表
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "body": {"method": "GET"}})
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": "m2"}, {"id": "m1"}]}).encode())

            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                if outer.fail_status:
                    self.send_response(outer.fail_status)
                    self.send_header("content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": {"message": "invalid api key"}}).encode())
                    return
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for i in range(0, len(outer.reply), 6):
                    chunk = {"choices": [{"delta": {"content": outer.reply[i:i + 6]}}]}
                    self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()

    def all_text(self) -> str:
        return json.dumps([r["body"] for r in self.requests], ensure_ascii=False)

    def user_text(self) -> str:
        """所有请求中 role=user 的消息正文（不含 JSON 键名和系统提示）。"""
        return "\n".join(m["content"] for r in self.requests for m in r["body"].get("messages", []) if m["role"] == "user")
