"""Pre-allocated array buffers for various calculation intensive processes"""
from enum import IntEnum, auto
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import DTypeLike

class BufferOverflowError(Exception):
    """Custom exception for buffer overflow situations"""
    pass

class ArrayInitializationStrategy(IntEnum):
    """Strategies for initializing array buffers"""
    Zeros = auto()
    Ones = auto()
    Empty = auto()

@dataclass(slots=True)
class ArrayBuffer:
    """Class representing an allocated array buffer"""
    size: int
    dtype: DTypeLike = field(init=True, default=np.float64)
    strategy: ArrayInitializationStrategy = field(init=True, default=ArrayInitializationStrategy.Empty)
    array: np.ndarray = field(init=False)
    current_idx: int = field(init=False, default=0)

    def __post_init__(self):
        """Initialize the buffer based on the specified strategy"""
        self.dtype = np.dtype(self.dtype)
        self.array = self._setup_array(self.size, self.dtype, self.strategy)

    @staticmethod
    def _setup_array(size, dtype, strategy) -> np.ndarray:
        if strategy == ArrayInitializationStrategy.Zeros:
            return np.zeros(size, dtype=dtype)
        elif strategy == ArrayInitializationStrategy.Ones:
            return np.ones(size, dtype=dtype)
        elif strategy == ArrayInitializationStrategy.Empty:
            return np.empty(size, dtype=dtype)
        else:
            raise ValueError(f"Unsupported initialization strategy: {strategy}")

    def get_slice(self, size: int) -> np.ndarray:
        """Get a slice of the buffer for writing data, ensuring it does not exceed allocated size"""
        if self.current_idx + size > self.size:
            raise BufferOverflowError(f"Requested slice size {size} exceeds remaining buffer capacity {self.size - self.current_idx}")
        arr = self.array[self.current_idx:self.current_idx + size]
        self.current_idx += size
        return arr

    def reset(self) -> None:
        self.current_idx = 0

@dataclass(slots=True)
class ExpandableArrayBuffer(ArrayBuffer):
    """Array buffer that can expand its size when needed"""
    expansion_factor: float = field(init=True, default=0.95)

    def __post_init__(self):
        """Initialize the buffer based on the specified strategy"""
        super().__post_init__()
        if not (0 < self.expansion_factor <= 1):
            raise ValueError(f"Expansion factor must be between 0 and 1, got {self.expansion_factor}")

    def get_slice(self, size: int) -> np.ndarray:
        """Get a slice of the buffer for writing data, expanding if necessary"""
        if self.current_idx + size > self.size:
            self._expand_buffer(max(self.size + size, int(self.size * (1 + self.expansion_factor))))
        arr = self.array[self.current_idx:self.current_idx + size]
        self.current_idx += size
        return arr

    def _expand_buffer(self, new_size: int) -> None:
        """Expand the buffer to a new size"""
        new_array = self._setup_array(new_size, self.dtype, self.strategy)
        new_array[:self.size] = self.array
        self.array = new_array
        self.size = new_size

class PaginatedArrayBuffer:
    """Array buffer that provides paginated access to its data"""
    page_size: int = field(init=True, default=100_000)
    dtype: DTypeLike = field(init=True, default=np.float64)
    strategy: ArrayInitializationStrategy = field(init=True, default=ArrayInitializationStrategy.Empty)
    pages: list[np.ndarray] = field(init=False, default_factory=list) # list of allocated pages
    page_sizes: list[int] = field(init=False, default_factory=list) # track size of each page
    page_cursors: list[int] = field(init=False, default_factory=list) # track current page index for writing
    current_page_idx: int = field(init=False, default=-1) # no pages allocated yet

    def __post_init__(self):
        self.dtype = np.dtype(self.dtype)

    @staticmethod
    def _setup_array(size: int, dtype: DTypeLike, strategy) -> np.ndarray:
        if strategy == ArrayInitializationStrategy.Zeros:
            return np.zeros(size, dtype=dtype)
        elif strategy == ArrayInitializationStrategy.Ones:
            return np.ones(size, dtype=dtype)
        elif strategy == ArrayInitializationStrategy.Empty:
            return np.empty(size, dtype=dtype)
        else:
            raise ValueError(f"Unsupported initialization strategy: {strategy}")

    def _allocate_new_page(self, size: int) -> None:
        actual_size = max(self.page_size, size)
        new_array = self._setup_array(actual_size, self.dtype, self.strategy)
        self.pages.append(new_array)
        self.page_sizes.append(actual_size)
        self.page_cursors.append(0)
        self.current_page_idx = len(self.pages) - 1

    def get_page(self, page: int) -> np.ndarray:
        """Get a specific page of the buffer"""
        if self.current_page_idx < 0:
            raise IndexError("Index out of range")
        return self.pages[page]

    @property
    def current_array_idx(self) -> int:
        """Get the current index in the current page for writing data"""
        if self.current_page_idx < 0:
            raise IndexError("No pages allocated yet")
        return self.page_cursors[self.current_page_idx]

    @property
    def current_page_size(self) -> int:
        """Get the size of the current page"""
        if self.current_page_idx < 0:
            raise IndexError("No pages allocated yet")
        return self.page_sizes[self.current_page_idx]

    @property
    def current_page(self):
        """Get the current page array for writing data"""
        if self.current_page_idx < 0:
            raise IndexError("No pages allocated yet")
        return self.pages[self.current_page_idx]

    def get_slice(self, size: int) -> np.ndarray:
        if self.current_page_idx < 0:
            self._allocate_new_page(size)

        current_cursor = self.current_page_idx
        current_page_size = self.current_page_size

        if current_cursor + size > current_page_size:
            self._allocate_new_page(size)

        page = self.current_page
        cursor = self.current_page_idx
        arr = page[cursor:cursor + size]
        self.page_cursors[self.current_page_idx] += size

        return arr

    def release_slice(self, size: int) -> None:
        if self.current_page_idx < 0:
            pass
        current_cursor = self.current_page_idx
        if size > current_cursor:
            self.page_cursors[self.current_page_idx] = 0
        else:
            self.page_cursors[self.current_page_idx] -= size


    def release_page(self, page: int) -> None:
        if 0 <= page < len(self.page_cursors):
            self.page_cursors[page] = 0


    def get_used_arrays(self) -> list[np.ndarray]:
        """Get a list of all currently used arrays across pages"""
        used_arrays = []
        for page, cursor in zip(self.pages, self.page_cursors):
            if cursor > 0:
                used_arrays.append(page[:cursor])
        return used_arrays

    def get_array_of_used_arrays(self) -> np.ndarray:
        arrs = self.get_used_arrays()
        if not arrs:
            return np.empty((0,), dtype=self.dtype)
        else:
            total_used = self.total_used
            result = np.empty(total_used, dtype=self.dtype)
            for arr in arrs:
                size = len(arr)
                result[:size] = arr
                result = result[size:]
            return result


    def reset(self) -> None:
        for i in range(len(self.page_cursors)):
            self.page_cursors[i] = 0
        if self.pages:
            self.current_page_idx = 0
        else:
            self.current_page_idx = -1

    def clear(self):
        self.pages.clear()
        self.page_sizes.clear().clear()
        self.page_cursors.clear()
        self.current_page_idx = -1

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def total_capacity(self) -> int:
        return sum(self.page_sizes)

    @property
    def total_used(self) -> int:
        return sum(self.page_cursors)