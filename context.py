from dataclasses import dataclass
from pathlib import Path


CTX_KEY = "ox_ctx"


@dataclass
class OxContext:
  root: Path

