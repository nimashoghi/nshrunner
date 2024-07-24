import logging
import sys
from collections import abc
from typing import Any

log = logging.getLogger(__name__)

_builtin_or_std_modules = sys.stdlib_module_names.union(sys.builtin_module_names)


def _resolve_leaf_module(module: str, ignore_builtin: bool):
    # First, resolve the module name into its first part. E.g.,:
    # mymodule.submodule.MyClass => mymodule
    module = module.split(".", 1)[0]

    # Ignore builtin or standard library modules
    if ignore_builtin and module in _builtin_or_std_modules:
        return ()

    return (module,)


def _resolve_modules_from_value(
    value: Any,
    visited: set[str],
    ignore_builtin: bool = True,
):
    """
    Resolve the modules from the given value.

    The core idea here that we want to take a list of given arguments to our runner, and resolve them into a list of modules that we want to snapshot.

    Example below:
    ```python
    # module: mymodule.submodule

    class MyClass: pass

    _resolve_modules_from_value("mymodule.submodule.MyClass") => []
    _resolve_modules_from_value(MyClass) => ["mymodule.submodule"]
    _resolve_modules_from_value(MyClass()) => ["mymodule.submodule"]
    _resolve_modules_from_value((MyClass(),)) => ["mymodule.submodule"]
    _resolve_modules_from_value(({"key": [MyClass()]},)) => ["mymodule.submodule"]

    ```
    """
    modules: set[str] = set()

    match value:
        # If type, resolve the module from the type
        # + all of its bases
        case type():
            modules.update(_resolve_leaf_module(value.__module__, ignore_builtin))
            for base in value.__bases__:
                modules.update(
                    _resolve_modules_from_value(
                        base,
                        visited,
                        ignore_builtin=ignore_builtin,
                    )
                )
        # If a collection, resolve the module from each item.
        # First, we handle mappings because we need to resolve
        # the modules from the keys and values separately.
        case abc.Mapping():
            for key, value in value.items():
                modules.update(
                    _resolve_modules_from_value(
                        key,
                        visited,
                        ignore_builtin=ignore_builtin,
                    )
                )
                modules.update(
                    _resolve_modules_from_value(
                        value,
                        visited,
                        ignore_builtin=ignore_builtin,
                    )
                )
        # Now, we handle any other collection
        case abc.Collection():
            for item in value:
                modules.update(
                    _resolve_modules_from_value(
                        item,
                        visited,
                        ignore_builtin=ignore_builtin,
                    )
                )
        # Anything else that has a "__module__" attribute
        case _ if hasattr(value, "__module__"):
            modules.update(_resolve_leaf_module(value.__module__, ignore_builtin))
            # Also process the parent type
            modules.update(
                _resolve_modules_from_value(
                    type(value),
                    visited,
                    ignore_builtin=ignore_builtin,
                )
            )
        # Anything else that doesn't have a "__module__" attribute -- we take the type
        # and resolve the module from that.
        case _:
            modules.update(
                _resolve_modules_from_value(
                    type(value),
                    visited,
                    ignore_builtin=ignore_builtin,
                )
            )

    return modules


def _resolve_parent_modules(configs: abc.Sequence[Any], ignore_builtin: bool = True):
    modules = set[str]()
    visited = set[str]()  # we can keep a global set of visited modules

    for config in configs:
        config_modules = _resolve_modules_from_value(
            config,
            visited,
            ignore_builtin=ignore_builtin,
        )
        if "__main__" in config_modules:
            log.warning(
                f"Config class (or a child class) {config.__class__.__name__} is defined in the main script.\n"
                "Snapshotting the main script is not supported.\n"
                "Skipping snapshotting of the config class's module."
            )
            config_modules.remove("__main__")

        modules.update(config_modules)

    return list(modules)
