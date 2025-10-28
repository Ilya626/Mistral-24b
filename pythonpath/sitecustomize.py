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

from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
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
    "num_layers",
    "n_layers",
    "n_layer",
    "hidden_layers",
    "num_transformer_layers",
    "num_decoder_layers",
)

_NON_NUM_HIDDEN_LAYER_CANDIDATES: Sequence[str] = tuple(
    name for name in _LAYER_COUNT_CANDIDATES if name != "num_hidden_layers"
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

    # ``Mistral3Config`` exposes ``num_layers`` as either a scalar, a
    # mapping, or a sequence depending on the code path that constructed the
    # config object.  For mappings we look for any positive integer values and
    # use the maximum – this matches how mergekit interprets layer ranges.  For
    # sequences we fall back to their length, but also attempt to coerce the
    # individual members in case the sequence simply wraps a single numeric
    # entry (e.g. ``[40]``).
    if isinstance(value, MappingABC):
        candidates: list[int] = []
        for item in value.values():
            coerced = _coerce_layer_count(item)
            if coerced is not None:
                candidates.append(coerced)
        if candidates:
            return max(candidates)
        return None

    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        candidates: list[int] = []
        for item in value:
            coerced = _coerce_layer_count(item)
            if coerced is not None:
                candidates.append(coerced)
        if candidates:
            return max(candidates)
        try:
            value = len(value)
        except Exception:
            return None

    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


def _layer_range_length(
    layer_range: Any, *, preferred: int | None = None
) -> int | None:
    if not isinstance(layer_range, SequenceABC):
        return None

    try:
        start = int(layer_range[0])
        stop = int(layer_range[1])
    except Exception:
        return None

    step = 1
    if len(layer_range) >= 3:
        try:
            candidate_step = int(layer_range[2])
            if candidate_step != 0:
                step = candidate_step
        except Exception:
            pass

    if step == 0:
        return None

    if step > 0:
        if stop < start:
            return None
        inclusive = ((stop - start) // step) + 1
        exclusive = ((stop - start) // step)
    else:
        if stop > start:
            return None
        step = -step
        inclusive = ((start - stop) // step) + 1
        exclusive = ((start - stop) // step)

    candidates = [value for value in (inclusive, exclusive) if value and value > 0]
    if not candidates:
        return None

    if preferred is not None:
        for candidate in candidates:
            if candidate == preferred:
                return candidate

    return max(candidates)


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


def _get_layer_count(
    config: Any, *, candidates: Sequence[str] | None = None
) -> int | None:
    lookup = candidates if candidates is not None else _LAYER_COUNT_CANDIDATES

    value = _resolve_config_value(config, lookup)
    coerced = _coerce_layer_count(value)
    if coerced is not None:
        return coerced

    visited: set[int] = set()
    layer_count = _search_layer_count(config, visited, lookup)
    if layer_count is not None:
        return layer_count

    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        try:
            data = to_dict()
        except Exception:
            data = None
        if isinstance(data, MappingABC):
            layer_count = _search_layer_count(data, visited, lookup)
            if layer_count is not None:
                return layer_count

    return None


def _search_layer_count(
    value: Any, visited: set[int], candidates: Sequence[str]
) -> int | None:
    if value is None:
        return None

    marker = id(value)
    if marker in visited:
        return None
    visited.add(marker)

    for candidate in candidates:
        if hasattr(value, candidate):
            try:
                attr_value = getattr(value, candidate)
            except Exception:
                attr_value = None
            coerced = _coerce_layer_count(attr_value)
            if coerced is not None:
                return coerced

    mapping: Mapping[Any, Any] | None = None
    if isinstance(value, MappingABC):
        mapping = value
    else:
        dictionary = getattr(value, "__dict__", None)
        if isinstance(dictionary, MappingABC):
            mapping = dictionary

    if mapping is not None:
        for candidate in candidates:
            if candidate in mapping:
                coerced = _coerce_layer_count(mapping[candidate])
                if coerced is not None:
                    return coerced
        for item in mapping.values():
            layer_count = _search_layer_count(item, visited, candidates)
            if layer_count is not None:
                return layer_count

    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            layer_count = _search_layer_count(item, visited, candidates)
            if layer_count is not None:
                return layer_count

    return None


def _ensure_layer_attribute(config: Any) -> int | None:
    """Ensure ``num_hidden_layers`` is present on ``config``.

    ``mergekit`` expects modern Mistral configs to expose their layer count
    via a ``num_hidden_layers`` attribute.  The newer ``Mistral3Config`` uses
    different names (``n_layers``/``num_layers``), which causes
    ``AttributeError`` during merge planning.  Rather than temporarily
    monkey-patching the attribute, populate it permanently for the lifetime of
    the config object so that later ``ArchitectureInfo`` helpers can read it
    without tripping over a missing attribute.
    """

    existing_value = None
    if hasattr(config, "num_hidden_layers"):
        try:
            existing_value = getattr(config, "num_hidden_layers")
        except Exception:
            existing_value = None

    existing_count = _coerce_layer_count(existing_value)

    derived_count = _get_layer_count(config, candidates=_NON_NUM_HIDDEN_LAYER_CANDIDATES)
    if derived_count is not None:
        if existing_count != derived_count:
            _set_config_attr(config, "num_hidden_layers", derived_count)
        return derived_count

    if existing_count is not None:
        return existing_count

    layer_count = _get_layer_count(config)
    if layer_count is None:
        return None
    if _set_config_attr(config, "num_hidden_layers", layer_count):
        return layer_count
    return layer_count


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

    original_num_layers = architecture.ArchitectureInfo.num_layers

    def patched_num_layers(self: Any, config: Any) -> int:
        layer_count = _ensure_layer_attribute(config)

        if layer_count is not None:
            return layer_count

        if hasattr(config, "num_hidden_layers"):
            try:
                return original_num_layers(self, config)
            except AttributeError:
                pass

        layer_count = _get_layer_count(config)
        if layer_count is not None:
            _set_config_attr(config, "num_hidden_layers", layer_count)
            return layer_count

        return original_num_layers(self, config)

    architecture.get_architecture_info = patched_get_architecture_info
    architecture.ArchitectureInfo.num_layers = patched_num_layers
    architecture._mistral3_patch_applied = True

    try:
        from mergekit import config as mk_config  # type: ignore
    except Exception:
        return

    ModelConfig = getattr(mk_config, "ModelConfig", None)
    if ModelConfig is None or getattr(ModelConfig, "_mistral3_patch_applied", False):
        return

    original_layer_count = ModelConfig.layer_count

    def patched_layer_count(self: Any) -> int:
        config_obj = getattr(self, "config", None)
        preferred = None
        if config_obj is not None:
            preferred = _ensure_layer_attribute(config_obj)

        layer_range = getattr(self, "layer_range", None)
        range_count = _layer_range_length(layer_range, preferred=preferred)
        if range_count is not None:
            return range_count

        if preferred is not None:
            return preferred

        return original_layer_count(self)

    ModelConfig.layer_count = patched_layer_count
    ModelConfig._mistral3_patch_applied = True


def _main() -> None:
    _patch_mergekit()


_main()
