"""stats_logger.py — Arka plan istatistik kaydı (append-only JSONL)."""
import json, time
from datetime import datetime
from pathlib import Path

class StatsLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir); self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "stats_log.jsonl"
        self.job = {}; self._timers = {}

    def start_job(self, video_path, profile, scope, content_type):
        self.job = {"job_id": f"job_{datetime.now():%Y%m%d_%H%M%S}",
                    "timestamp": datetime.now().isoformat(),
                    "video": video_path, "profile": profile,
                    "scope": scope, "type": content_type, "stages": {}, "errors": []}

    def start_stage(self, name): self._timers[name] = time.time()
    def end_stage(self, name, status="ok", **kw):
        d = {"sec": round(time.time()-self._timers.get(name,time.time()),2), "status": status}
        d.update(kw); self.job["stages"][name] = d

    def log_error(self, e): self.job.setdefault("errors",[]).append(str(e))
    def finish_job(self, total_sec=None):
        if total_sec: self.job["total_sec"] = round(total_sec,2)
        try:
            with open(self.log_file,"a",encoding="utf-8") as f:
                f.write(json.dumps(self.job, ensure_ascii=False)+"\n")
        except Exception: pass
