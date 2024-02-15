"""Types for NeuralFlow."""

from typing import Union, Tuple, List, Any

# define a type alias for Jax Pytrees
Pytree = Union[Tuple, List]
Bijector_Info = Tuple[str, tuple]
Array = Any
