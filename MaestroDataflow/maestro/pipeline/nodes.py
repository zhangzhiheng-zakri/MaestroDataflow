"""
Node classes for building pipeline data flow graphs.
"""
from __future__ import annotations
from typing import List, Optional, Union, Dict, Any

from maestro.core import OperatorABC
from maestro.utils.storage import MaestroStorage


class KeyNode:
    """
    KeyNode represents a node in the pipeline data flow graph.
    It tracks the flow of data keys between operators.
    """
    def __init__(
        self,
        key_para_name: str,
        key: str,
        ptr: Optional[List[KeyNode]] = None
    ):
        """
        Initialize a KeyNode.

        Args:
            key_para_name: Name of the parameter in the operator's run function
            key: The key name in the storage
            ptr: Pointers to next KeyNode(s), used to build a list of keys
        """
        self.key_para_name = key_para_name  # name of the parameter in the operator's run function
        self.key = key
        self.ptr = ptr if ptr is not None else []  # ptr to next KeyNode(s)
        self.index = -1  # Will be set when added to a pipeline

    def set_index(self, index: int) -> None:
        """Set the index of this node in the pipeline."""
        self.index = index

    def __str__(self) -> str:
        """String representation of the KeyNode."""
        current_id = hex(id(self))
        ptr_status = [
            (node.key, node.index, hex(id(node))) for node in self.ptr
        ] if len(self.ptr) != 0 else ["None"]
        ptr_str = "".join([
            f"\n      <{item}>" for item in ptr_status
        ])
        return f"\n    KeyNode[{current_id}](key_para_name={self.key_para_name}, key={self.key}, ptr_keys={ptr_str})"

    def __repr__(self) -> str:
        """Representation of the KeyNode."""
        return self.__str__()