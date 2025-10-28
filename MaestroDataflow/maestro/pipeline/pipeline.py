"""
Core Pipeline implementation for MaestroDataflow.
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional, Type, Union, Callable
import inspect
from abc import ABC, abstractmethod

from maestro.core import OperatorABC
from maestro.utils.storage import MaestroStorage
from .nodes import KeyNode


class PipelineABC(ABC):
    """
    Abstract base class for all pipelines in MaestroDataflow.
    Defines the interface that all pipeline implementations must follow.
    """

    @abstractmethod
    def add_operator(self, operator: OperatorABC, name: str) -> None:
        """
        Add an operator to the pipeline.

        Args:
            operator: The operator to add
            name: A unique name for the operator
        """
        pass

    @abstractmethod
    def connect(self, from_op: str, to_op: str,
                from_key: str, to_key: str,
                to_param: str) -> None:
        """
        Connect two operators in the pipeline.

        Args:
            from_op: Name of the source operator
            to_op: Name of the destination operator
            from_key: Output key from the source operator
            to_key: Input key for the destination operator
            to_param: Parameter name in the destination operator's run method
        """
        pass

    @abstractmethod
    def run(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the pipeline.

        Args:
            input_data: Optional input data to start the pipeline

        Returns:
            The output data from the pipeline
        """
        pass


class Pipeline(PipelineABC):
    """
    Standard implementation of a data processing pipeline.
    Manages operators and their connections, and handles data flow between them.
    """

    def __init__(self, storage: MaestroStorage):
        """
        Initialize a new Pipeline.

        Args:
            storage: Optional storage to use for this pipeline. If not provided,
                    a new FileStorage will be created.
        """
        self.operators: Dict[str, OperatorABC] = {}
        self.connections: Dict[str, List[KeyNode]] = {}
        self.execution_order: List[str] = []
        self.storage = storage

    def add_operator(self, operator: OperatorABC, name: str) -> None:
        """
        Add an operator to the pipeline.

        Args:
            operator: The operator to add
            name: A unique name for the operator

        Raises:
            ValueError: If an operator with the same name already exists
        """
        if name in self.operators:
            raise ValueError(f"Operator with name '{name}' already exists in the pipeline")

        self.operators[name] = operator
        self.connections[name] = []
        self.execution_order.append(name)

    def connect(self, from_op: str, to_op: str,
                from_key: str, to_key: str,
                to_param: str) -> None:
        """
        Connect two operators in the pipeline.

        Args:
            from_op: Name of the source operator
            to_op: Name of the destination operator
            from_key: Output key from the source operator
            to_key: Input key for the destination operator
            to_param: Parameter name in the destination operator's run method

        Raises:
            ValueError: If either operator does not exist in the pipeline
        """
        if from_op not in self.operators:
            raise ValueError(f"Source operator '{from_op}' does not exist in the pipeline")
        if to_op not in self.operators:
            raise ValueError(f"Destination operator '{to_op}' does not exist in the pipeline")

        # Create a KeyNode for this connection
        node = KeyNode(to_param, to_key)

        # Add the node to the connections for the source operator
        self.connections[from_op].append(node)

        # Update execution order to ensure dependencies are met
        self._update_execution_order(from_op, to_op)

    def _update_execution_order(self, from_op: str, to_op: str) -> None:
        """
        Update the execution order to ensure dependencies are met.

        Args:
            from_op: Source operator name
            to_op: Destination operator name
        """
        # If to_op is already before from_op in the execution order, we need to reorder
        if self.execution_order.index(to_op) < self.execution_order.index(from_op):
            # Remove to_op from its current position
            self.execution_order.remove(to_op)
            # Insert to_op after from_op
            from_index = self.execution_order.index(from_op)
            self.execution_order.insert(from_index + 1, to_op)

    def run(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        运行Pipeline，逐个执行添加的操作符。
        返回每个操作符的执行元信息。
        """
        results: Dict[str, Any] = {}

        for op_name in self.execution_order:
            operator = self.operators[op_name]
            run_args: Dict[str, Any] = {"storage": self.storage}
            for connection in self.connections.get(op_name, []):
                run_args[connection.key_para_name] = connection.key
            meta = operator.run(**run_args)
            results[op_name] = meta

        return results


class BatchPipeline(Pipeline):
    """
    A pipeline that processes data in batches.
    Useful for handling large datasets that don't fit in memory.
    """

    def __init__(self, storage: Optional[MaestroStorage] = None, batch_size: int = 100):
        """
        Initialize a new BatchPipeline.

        Args:
            storage: Optional storage to use for this pipeline
            batch_size: Number of items to process in each batch
        """
        super().__init__(storage)
        self.batch_size = batch_size

    def run(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the pipeline in batches.

        Args:
            input_data: Optional input data to start the pipeline

        Returns:
            The output data from the pipeline
        """
        # Initialize storage with input data if provided
        if input_data is not None:
            for key, value in input_data.items():
                # Check if the value is a list that needs batching
                if isinstance(value, list) and len(value) > self.batch_size:
                    # Process in batches
                    results = []
                    for i in range(0, len(value), self.batch_size):
                        batch = value[i:i + self.batch_size]

                        # Reset storage for this batch
                        self.storage.reset()
                        self.storage.write(key, batch)

                        # Run the pipeline for this batch
                        batch_result = super().run()

                        # Collect results
                        for result_key, result_value in batch_result.items():
                            if result_key not in results:
                                results[result_key] = []
                            results[result_key].extend(result_value)

                    # Return combined results
                    return results
                else:
                    # For small data, just use the regular pipeline
                    self.storage.write(key, value)

        # For non-batched data, use the regular pipeline run
        return super().run()