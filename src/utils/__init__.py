import hashlib
import time
from dataclasses import dataclass
from typing import Any


def hash_str(str_: str) -> int:
    sha256_hash = hashlib.sha256(str_.encode()).digest()
    hash_int = int.from_bytes(sha256_hash, byteorder="big")
    return hash_int


class Timer:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time


@dataclass
class ResultWithTime:
    result: Any
    time: float

    def __iter__(self):
        return iter((self.result, self.time))
