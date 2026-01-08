from __future__ import annotations

# Compatibility wrapper expected by setup.sh / README
# Real implementation lives in replay_csv.py

from .replay_csv import main


if __name__ == "__main__":
    main()
