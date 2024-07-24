import functools
import logging
import sys
from collections import abc
from typing import Any

log = logging.getLogger(__name__)

_builtin_or_std_modules = sys.stdlib_module_names.union(sys.builtin_module_names)


@functools.cache
def _resolve_leaf_module(module: str, ignore_builtin: bool):
    # First, resolve the module name into its first part. E.g.,:
    # mymodule.submodule.MyClass => mymodule
    module = module.split(".", 1)[0]

    # Ignore builtin or standard library modules
    if ignore_builtin and module in _builtin_or_std_modules:
        return ()

    return (module,)


def _resolve_type(
    type_: type,
    cached: dict[str, set[str]],
    ignore_builtin: bool = True,
    deep: bool = False,
):
    if (cached_modules := cached.get(type_.__module__)) is not None:
        return cached_modules

    modules = set[str]()

    # Resolve the module of the type
    modules.update(_resolve_leaf_module(type_.__module__, ignore_builtin))

    # Resolve the module of the type's bases
    if deep:
        for base in type_.__bases__:
            modules.update(_resolve_type(base, cached, ignore_builtin, deep))

    cached[type_.__module__] = modules
    return modules


def _filter_container_shallow(value: abc.Collection | abc.Mapping):
    # When deep=False, we only look inside containers if they are built-in types.
    # In other words, whenever we hit a user-defined class, we stop looking inside it.

    return _resolve_leaf_module(type(value).__module__, ignore_builtin=True) == ()


def _resolve_modules_from_value(
    value: Any,
    cached: dict[str, set[str]],
    ignore_builtin: bool = True,
    deep: bool = False,
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
    is_type = False

    match value:
        # If type, resolve the module from the type
        # + all of its bases
        case type():
            modules.update(_resolve_type(value, cached, ignore_builtin, deep))
            is_type = True
        # If a primitive python type, just process the value
        case (
            int()
            | float()
            | str()
            | bool()
            | complex()
            | bytes()
            | bytearray()
            | None
        ):
            pass
        # If a collection, resolve the module from each item.
        # First, we handle mappings because we need to resolve
        # the modules from the keys and values separately.
        case abc.Mapping() if deep or _filter_container_shallow(value):
            for key, value in value.items():
                modules.update(
                    _resolve_modules_from_value(
                        key,
                        cached,
                        ignore_builtin=ignore_builtin,
                    )
                )
                modules.update(
                    _resolve_modules_from_value(
                        value,
                        cached,
                        ignore_builtin=ignore_builtin,
                    )
                )
        # Now, we handle any other collection
        case abc.Collection() if deep or _filter_container_shallow(value):
            for item in value:
                modules.update(
                    _resolve_modules_from_value(
                        item,
                        cached,
                        ignore_builtin=ignore_builtin,
                    )
                )
        # Anything else that has a "__module__" attribute
        case _ if hasattr(value, "__module__"):
            modules.update(_resolve_leaf_module(value.__module__, ignore_builtin))
        case _:
            pass

    # We should also resolve the type of the value, if it's not a type itself
    if not is_type:
        modules.update(
            _resolve_modules_from_value(
                type(value),
                cached,
                ignore_builtin=ignore_builtin,
            )
        )

    return modules


def _resolve_parent_modules(
    configs: abc.Sequence[Any],
    ignore_builtin: bool = True,
    deep: bool = False,
):
    modules = set[str]()
    cached = dict[str, set[str]]()  # we can keep a global set of visited modules

    for config in configs:
        config_modules = _resolve_modules_from_value(
            config,
            cached,
            ignore_builtin=ignore_builtin,
            deep=deep,
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
