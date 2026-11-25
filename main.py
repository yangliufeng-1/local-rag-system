"""无需安装包即可运行的入口：python main.py <command> [options]"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_system.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
