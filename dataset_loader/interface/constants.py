import os

HOME = os.getenv("HOME")

# Types
Task = str

# Defaults
DEFAULT_PATH = f"{HOME}/.datasets"

__all__ = ["Task"]
