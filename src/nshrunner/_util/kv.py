from __future__ import annotations

import pickle
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from flufl.lock import Lock, LockError
from typing_extensions import override


class GlobalKVStore(MutableMapping[str, Any]):
    """A persistent key-value store using Pickle with NFS-safe locking, following MutableMapping."""

    @override
    def __init__(
        self,
        filepath: str | Path,
        protocol: int = 4,
    ):
        """Initialize the key-value store with a file path and Pickle protocol.
        Args:
            filepath: Path to the Pickle file where data will be stored.
            protocol: Pickle protocol version to use (default is 4).
        """
        self.filepath = Path(filepath)
        self.protocol = protocol
        self.lock = Lock(str(self.filepath.with_suffix(".lock")), default_timeout=60)

    def _load(self):
        """Load the Pickle file, returning an empty dict if it doesn't exist."""
        if self.filepath.exists():
            with open(self.filepath, "rb") as f:
                return pickle.load(f)
        return {}

    def _save(self, data):
        """Save the data to the Pickle file."""
        with open(self.filepath, "wb") as f:
            pickle.dump(data, f, protocol=self.protocol)

    @override
    def __getitem__(self, key: str):
        """Retrieve a value by key, raising KeyError if not found."""
        if not isinstance(key, str):
            raise TypeError("Keys must be strings")
        try:
            with self.lock:
                data = self._load()
                if key not in data:
                    raise KeyError(key)
                return data[key]
        except LockError as e:
            raise RuntimeError(f"Failed to acquire lock for get operation: {e}")

    @override
    def __setitem__(self, key: str, value: Any):
        """Set a key-value pair in the store."""
        if not isinstance(key, str):
            raise TypeError("Keys must be strings")
        try:
            with self.lock:
                data = self._load()
                data[key] = value
                self._save(data)
        except LockError as e:
            raise RuntimeError(f"Failed to acquire lock for set operation: {e}")
        except pickle.PicklingError as e:
            raise ValueError(f"Failed to serialize value for key '{key}': {e}")

    @override
    def __delitem__(self, key: str):
        """Delete a key from the store, raising KeyError if not found."""
        if not isinstance(key, str):
            raise TypeError("Keys must be strings")
        try:
            with self.lock:
                data = self._load()
                if key not in data:
                    raise KeyError(key)
                del data[key]
                self._save(data)
        except LockError as e:
            raise RuntimeError(f"Failed to acquire lock for delete operation: {e}")

    @override
    def __iter__(self):
        """Return an iterator over the keys in the store."""
        try:
            with self.lock:
                data = self._load()
                return iter(data)
        except LockError as e:
            raise RuntimeError(f"Failed to acquire lock for iter operation: {e}")

    @override
    def __len__(self):
        """Return the number of key-value pairs in the store."""
        try:
            with self.lock:
                data = self._load()
                return len(data)
        except LockError as e:
            raise RuntimeError(f"Failed to acquire lock for len operation: {e}")
