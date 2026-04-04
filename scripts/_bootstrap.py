from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable


def _configure_utf8_stdio() -> None:
    """Normalize process text I/O to UTF-8 across Windows and Unix-like systems."""

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")

    for stream_name in ("stdout", "stderr", "stdin"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            kwargs = {"encoding": "utf-8", "errors": "replace"}
            if stream_name in {"stdout", "stderr"}:
                kwargs["line_buffering"] = True
            stream.reconfigure(**kwargs)
        except Exception:
            # Some embedded terminals do not expose a mutable text stream.
            continue


def bootstrap() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    _configure_utf8_stdio()


def run_cli(main_fn: Callable[[], None]) -> None:
    try:
        main_fn()
    except KeyboardInterrupt:
        print("执行已中断。", file=sys.stderr)
        raise SystemExit(130)
    except FileNotFoundError as exc:
        print(f"文件不存在: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except RuntimeError as exc:
        message = str(exc)
        if "torch.cuda.is_available()" in message or "请求设备" in message:
            print(
                f"设备错误: {message}\n如需临时改用 CPU，请追加参数 --set runtime.device=cpu",
                file=sys.stderr,
            )
            raise SystemExit(1)
        raise
