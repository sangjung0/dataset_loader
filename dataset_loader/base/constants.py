import os

HOME = os.getenv("HOME")

# Defaults
DEFAULT_PATH = f"{HOME}/.datasets"

__all__ = ["DEFAULT_PATH"]
