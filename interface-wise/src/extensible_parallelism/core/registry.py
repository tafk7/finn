"""
Operation Registry for the extensible parallelism system.

Provides a centralized registry for all supported parallelizable operations,
enabling dynamic discovery and instantiation.
"""

from typing import Dict, Type, List, Optional, Any
from ..core.base import ParallelizableOperation


class OperationRegistry:
    """
    Central registry for all supported parallelizable operations.

    This registry allows operations to be registered by name and instantiated
    dynamically, enabling extensibility and modularity.
    """

    _operations: Dict[str, Type[ParallelizableOperation]] = {}
    _categories: Dict[str, List[str]] = {}

    @classmethod
    def register(cls, op_type: str,
                 operation_class: Type[ParallelizableOperation],
                 category: str = "general") -> None:
        """
        Register a new operation type.

        Args:
            op_type: Unique identifier for the operation type
            operation_class: Class implementing ParallelizableOperation
            category: Category for organization (e.g., "matrix", "activation")
        """
        if not issubclass(operation_class, ParallelizableOperation):
            raise ValueError(f"Operation class must inherit from ParallelizableOperation")

        if op_type in cls._operations:
            raise ValueError(f"Operation type '{op_type}' already registered")

        cls._operations[op_type] = operation_class

        if category not in cls._categories:
            cls._categories[category] = []
        cls._categories[category].append(op_type)

    @classmethod
    def unregister(cls, op_type: str) -> None:
        """Unregister an operation type."""
        if op_type not in cls._operations:
            raise ValueError(f"Operation type '{op_type}' not registered")

        # Remove from operations
        del cls._operations[op_type]

        # Remove from categories
        for category_ops in cls._categories.values():
            if op_type in category_ops:
                category_ops.remove(op_type)

    @classmethod
    def create(cls, op_type: str, name: str, **kwargs) -> ParallelizableOperation:
        """
        Create an instance of a registered operation.

        Args:
            op_type: Type of operation to create
            name: Unique name for this operation instance
            **kwargs: Additional arguments for operation constructor

        Returns:
            Instance of the requested operation type
        """
        if op_type not in cls._operations:
            available = list(cls._operations.keys())
            raise ValueError(f"Unknown operation type '{op_type}'. Available types: {available}")

        operation_class = cls._operations[op_type]
        return operation_class(name=name, **kwargs)

    @classmethod
    def list_operations(cls) -> List[str]:
        """List all registered operation types."""
        return list(cls._operations.keys())

    @classmethod
    def list_categories(cls) -> Dict[str, List[str]]:
        """List all operation categories and their operations."""
        return dict(cls._categories)

    @classmethod
    def get_operations_by_category(cls, category: str) -> List[str]:
        """Get all operations in a specific category."""
        return cls._categories.get(category, [])

    @classmethod
    def get_operation_class(cls, op_type: str) -> Type[ParallelizableOperation]:
        """Get the class for a registered operation type."""
        if op_type not in cls._operations:
            raise ValueError(f"Unknown operation type '{op_type}'")
        return cls._operations[op_type]

    @classmethod
    def is_registered(cls, op_type: str) -> bool:
        """Check if an operation type is registered."""
        return op_type in cls._operations

    @classmethod
    def get_operation_info(cls, op_type: str) -> Dict[str, Any]:
        """Get information about a registered operation."""
        if op_type not in cls._operations:
            raise ValueError(f"Unknown operation type '{op_type}'")

        operation_class = cls._operations[op_type]
        category = None

        # Find category
        for cat, ops in cls._categories.items():
            if op_type in ops:
                category = cat
                break

        return {
            "type": op_type,
            "class": operation_class.__name__,
            "module": operation_class.__module__,
            "category": category,
            "docstring": operation_class.__doc__ or "No documentation available"
        }

    @classmethod
    def clear(cls) -> None:
        """Clear all registered operations (mainly for testing)."""
        cls._operations.clear()
        cls._categories.clear()


def register_operation(op_type: str, category: str = "general"):
    """
    Decorator for registering operations.

    Usage:
        @register_operation("MatrixVector", category="matrix")
        class MatrixVectorOperation(ParallelizableOperation):
            ...
    """
    def decorator(operation_class: Type[ParallelizableOperation]):
        OperationRegistry.register(op_type, operation_class, category)
        return operation_class
    return decorator


# Auto-registration of built-in operations
def _register_builtin_operations():
    """Register built-in operations when module is imported."""
    # This will be called after all operation modules are defined
    pass
