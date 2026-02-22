"""
vram_manager.py — GPU bellek yönetimi.

Her stage sonrası model boşaltılır:
  del model → gc.collect() → torch.cuda.empty_cache()

VRAM bütçesi (sıralı, aynı anda tek model):
  [B] DeepFilterNet  → ~200 MB
  [C] PyAnnote       → ~1-2 GB
  [D] WhisperX       → ~3-4 GB
  [E] Ollama         → harici süreç (kendi VRAM yönetimi)
"""

import gc


class VRAMManager:
    """GPU bellek yöneticisi — stage'ler arası model boşaltma."""

    @staticmethod
    def release():
        """GPU belleğini serbest bırak."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass

    @staticmethod
    def get_usage() -> str:
        """Mevcut GPU bellek kullanımını döndür."""
        try:
            import torch
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_mem / 1024**3
                return f"{used:.1f}GB used / {reserved:.1f}GB reserved / {total:.1f}GB total"
        except ImportError:
            pass
        return "CPU (torch yok)"

    @staticmethod
    def get_device() -> str:
        """Kullanılabilir device döndür: 'cuda' veya 'cpu'."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
