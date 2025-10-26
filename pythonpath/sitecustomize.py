"""Runtime patches for external tools.

This module is automatically imported by Python when present on the
PYTHONPATH.  We use it to patch mergekit so that it recognises the
``Mistral3ForConditionalGeneration`` and ``MistralForConditionalGeneration``
architecture identifiers emitted by recent Mistral models.  The upstream
version of mergekit has not been updated for these names yet, so without
the patch mergekit aborts with errors such as
``RuntimeError: Unsupported architecture MistralForConditionalGeneration``.

The patch remaps these new identifiers to the older
``MistralForCausalLM`` handler, which is compatible with the model
structure used by the Vistral and Cydonia checkpoints.  This keeps
mergekit working without requiring a fork of the dependency or manual
edits in the environment.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence


def _normalise_architectures(value: Any) -> Sequence[str]:
    """Return a normalised tuple of architecture names from a config.

    Hugging Face configs expose ``architectures`` as either a list, a
    tuple, or ``None``.  Mergekit only ever looks at the first entry, but
    for our patch we treat any sequence defensively and coerce it into a
    tuple.  Unknown values (including ``None``) simply yield an empty
    tuple.
    """

    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return ()


_LAYER_COUNT_CANDIDATES: Sequence[str] = (
    "num_hidden_layers",
    "num_hidden_layers",
    "num_layers",
    "n_layers",
    "hidden_layers",
    "num_transformer_layers",
)


def _set_config_attr(config: Any, name: str, value: Any) -> bool:
    try:
        setattr(config, name, value)
        return True
    except Exception:
        try:
            object.__setattr__(config, name, value)
            return True
        except Exception:
            return False


def _del_config_attr(config: Any, name: str) -> None:
    try:
        delattr(config, name)
    except Exception:
        try:
            object.__delattr__(config, name)
        except Exception:
            pass


def _resolve_config_value(config: Any, names: Sequence[str]) -> Any:
    for candidate in names:
        if hasattr(config, candidate):
            try:
                value = getattr(config, candidate)
            except Exception:
                continue
            if value is not None:
                return value
    config_dict = getattr(config, "__dict__", None)
    if isinstance(config_dict, Mapping):
        for candidate in names:
            if candidate in config_dict:
                value = config_dict[candidate]
                if value is not None:
                    return value
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        try:
            data = to_dict()
        except Exception:
            data = None
        if isinstance(data, Mapping):
            for candidate in names:
                if candidate in data:
                    value = data[candidate]
                    if value is not None:
                        return value
    return None


def _coerce_layer_count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = len(value)
        except Exception:
            return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


def _push_patch(config: Any, restorers: list[Callable[[], None]], name: str, value: Any) -> bool:
    had_attr = hasattr(config, name)
    original: Any = None
    if had_attr:
        try:
            original = getattr(config, name)
        except Exception:
            had_attr = False
    if not _set_config_attr(config, name, value):
        return False

    def restore() -> None:
        try:
            if had_attr:
                _set_config_attr(config, name, original)
            else:
                _del_config_attr(config, name)
        except Exception:
            pass

    restorers.append(restore)
    return True


def _ensure_layer_attribute(config: Any) -> None:
    """Ensure ``num_hidden_layers`` is present on ``config``.

    ``mergekit`` expects modern Mistral configs to expose their layer count
    via a ``num_hidden_layers`` attribute.  The newer ``Mistral3Config`` uses
    different names (``n_layers``/``num_layers``), which causes
    ``AttributeError`` during merge planning.  Rather than temporarily
    monkey-patching the attribute, populate it permanently for the lifetime of
    the config object so that later ``ArchitectureInfo`` helpers can read it
    without tripping over a missing attribute.
    """

    if hasattr(config, "num_hidden_layers"):
        return
    value = _resolve_config_value(config, _LAYER_COUNT_CANDIDATES)
    layer_count = _coerce_layer_count(value)
    if layer_count is None:
        return
    _set_config_attr(config, "num_hidden_layers", layer_count)


def _patch_mergekit() -> None:
    try:
        from mergekit import architecture  # type: ignore
    except Exception:  # pragma: no cover - mergekit may not be installed locally
        return

    if getattr(architecture, "_mistral3_patch_applied", False):
        return

    original_get_architecture_info = architecture.get_architecture_info

    architecture_aliases: Mapping[str, str] = {
        "Mistral3ForConditionalGeneration": "MistralForCausalLM",
        "MistralForConditionalGeneration": "MistralForCausalLM",
    }

    def patched_get_architecture_info(config: Any) -> Any:
        architectures = _normalise_architectures(getattr(config, "architectures", ()))
        alias = architecture_aliases.get(architectures[0]) if architectures else None
        restorers: list[Callable[[], None]] = []
        try:
            if alias:
                _push_patch(config, restorers, "architectures", (alias,))
            _ensure_layer_attribute(config)
            return original_get_architecture_info(config)
        finally:
            while restorers:
                restore = restorers.pop()
                try:
                    restore()
                except Exception:
                    pass

    architecture.get_architecture_info = patched_get_architecture_info
    architecture._mistral3_patch_applied = True


def _main() -> None:
    _patch_mergekit()


_main()
