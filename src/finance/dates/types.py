from typing import TypeVar, Annotated
from finance.dates import Date
import numpy as np
import numpy.typing as npt

# define common types used in date module

# integer types
IntScalar = int | np.integer
IntArray = npt.NDArray[np.integer]
IntNpType = np.integer | IntArray
IntType = TypeVar("IntType", "IntScalar", "IntArray")

# date types
DateNp = Annotated[np.datetime64, "datetime64[D]"]
DateScalar = DateNp | Date
DateArray = Annotated[npt.NDArray[np.datetime64], "datetime64[D]"]
DateNpType = DateNp | DateArray
DateType = TypeVar("DateType", "DateScalar", "DateArray")