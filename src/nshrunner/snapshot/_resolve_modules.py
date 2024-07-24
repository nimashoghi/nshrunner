import logging
import sys
from collections import abc
from typing import Any

import nshutils

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
    value: Any, ignore_builtin: bool = True
) -> abc.Generator[str, None, None]:
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
    with nshutils.snoop():
        match value:
            # If type, resolve the module from the type
            # + all of its bases
            case type():
                yield from _resolve_leaf_module(value.__module__, ignore_builtin)
                for base in value.__bases__:
                    yield from _resolve_modules_from_value(base, ignore_builtin)
            # If a collection, resolve the module from each item.
            # First, we handle mappings because we need to resolve
            # the modules from the keys and values separately.
            case abc.Mapping():
                for key, value in value.items():
                    yield from _resolve_modules_from_value(key, ignore_builtin)
                    yield from _resolve_modules_from_value(value, ignore_builtin)
            # Now, we handle any other collection
            case abc.Collection():
                for item in value:
                    yield from _resolve_modules_from_value(item, ignore_builtin)
            # Anything else that has a "__module__" attribute
            case has_module_value if hasattr(has_module_value, "__module__"):
                yield from _resolve_leaf_module(
                    has_module_value.__module__, ignore_builtin
                )
                # Also process the parent type
                yield from _resolve_modules_from_value(type(value), ignore_builtin)
            # Anything else that doesn't have a "__module__" attribute -- we take the type
            # and resolve the module from that.
            case _:
                yield from _resolve_modules_from_value(type(value), ignore_builtin)


def _resolve_parent_modules(configs: abc.Sequence[Any], ignore_builtin: bool = True):
    modules_set = set[str]()

    for config in configs:
        for module in _resolve_modules_from_value(config, ignore_builtin):
            # NOTE: We also must handle the case where the config
            #   class's module is "__main__" (i.e. the config class
            #   is defined in the main script).
            if module == "__main__":
                log.warning(
                    f"Config class (or a child class) {config.__class__.__name__} is defined in the main script.\n"
                    "Snapshotting the main script is not supported.\n"
                    "Skipping snapshotting of the config class's module."
                )
                continue

            # Make sure to get the root module
            modules_set.add(module)

    return list(modules_set)
