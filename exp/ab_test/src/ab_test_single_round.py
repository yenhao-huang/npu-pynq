"""Compatibility entry point; defaults to one round, accepts --rounds N."""
if __package__:
    from .main import main
else:
    import sys
    from pathlib import Path
    source_dir = Path(__file__).resolve().parent
    # Keep the package-local logging module out of top-level import lookup.
    sys.path[:] = [entry for entry in sys.path if Path(entry).resolve() != source_dir]
    sys.path.insert(0, str(source_dir.parent))
    from src.main import main

if __name__ == "__main__":
    main()
