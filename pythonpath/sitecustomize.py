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
import os
from pathlib import Path
import sys
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

_LAYER_OVERRIDE_ATTR = "_mistral_layer_range_override"
_LAYER_DEBUG_FLAG = os.environ.get("MISTRAL_LAYER_DEBUG", "")


_TOKENIZER_SLOW_CANDIDATES: Sequence[tuple[str, str]] = (
    ("transformers.models.mistral3.tokenization_mistral3", "Mistral3Tokenizer"),
    ("transformers.models.mistral.tokenization_mistral", "MistralTokenizer"),
    ("transformers.models.llama.tokenization_llama", "LlamaTokenizer"),
)

_TOKENIZER_FAST_CANDIDATES: Sequence[tuple[str, str]] = (
    ("transformers.models.mistral3.tokenization_mistral3_fast", "Mistral3TokenizerFast"),
    ("transformers.models.mistral.tokenization_mistral_fast", "MistralTokenizerFast"),
    ("transformers.models.llama.tokenization_llama_fast", "LlamaTokenizerFast"),
)


def _normalise_override_key(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(Path(value).expanduser().resolve())
    except Exception:
        try:
            return os.path.abspath(os.path.expanduser(value))
        except Exception:
            return value


_LAYER_OVERRIDE_MAP: dict[str, int] = {}
_override_spec = os.environ.get("MISTRAL_LAYER_RANGE_OVERRIDES", "")
for raw_entry in _override_spec.split(":"):
    entry = raw_entry.strip()
    if not entry:
        continue
    path, sep, value = entry.partition("=")
    if not sep:
        continue
    try:
        count = int(value)
    except Exception:
        continue
    key = _normalise_override_key(path)
    if key is None:
        continue
    if count > 0:
        _LAYER_OVERRIDE_MAP[key] = count


def _log_layer_debug(message: str) -> None:
    if not _LAYER_DEBUG_FLAG:
        return
    try:
        print(f"[mistral-layer-debug] {message}", file=sys.stderr)
    except Exception:
        pass


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

    forward = step > 0
    step = abs(step)

    delta = stop - start
    if (forward and delta < 0) or (not forward and delta > 0):
        return None

    delta = abs(delta)
    exclusive = (delta + step - 1) // step

    inclusive: int | None = None
    if delta % step == 0:
        inclusive = exclusive + 1 if exclusive > 0 else 1

    candidates = []
    if exclusive > 0:
        candidates.append(exclusive)
    if inclusive is not None and inclusive != exclusive:
        candidates.append(inclusive)

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

    override_value = getattr(config, _LAYER_OVERRIDE_ATTR, None)
    override_count = _coerce_layer_count(override_value)

    if override_count is None:
        for candidate_attr in ("_name_or_path", "name_or_path"):
            candidate_value = getattr(config, candidate_attr, None)
            if isinstance(candidate_value, str):
                mapped = _LAYER_OVERRIDE_MAP.get(_normalise_override_key(candidate_value), None)
                if mapped is not None:
                    override_count = mapped
                    _set_config_attr(config, _LAYER_OVERRIDE_ATTR, override_count)
                    break

    existing_value = None
    if hasattr(config, "num_hidden_layers"):
        try:
            existing_value = getattr(config, "num_hidden_layers")
        except Exception:
            existing_value = None

    existing_count = _coerce_layer_count(existing_value)

    if override_count is not None:
        if existing_count != override_count:
            _set_config_attr(config, "num_hidden_layers", override_count)
        return override_count

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


def _import_first_available(candidates: Sequence[tuple[str, str]]) -> tuple[type | None, str | None]:
    for module_name, attr_name in candidates:
        try:
            module = __import__(module_name, fromlist=[attr_name])
        except Exception:
            module = None
        if module is None:
            continue
        candidate = getattr(module, attr_name, None)
        if candidate is not None:
            return candidate, attr_name
    return None, None


def _patch_sentencepiece_loader() -> None:
    try:
        import sentencepiece as spm  # type: ignore
    except Exception:
        return

    processor_cls = getattr(spm, "SentencePieceProcessor", None)
    if processor_cls is None:
        return

    if getattr(spm, "_mistral_sentencepiece_patch_applied", False):
        return

    original_load = getattr(processor_cls, "LoadFromFile", None)
    if not callable(original_load):
        return

    def patched_load(self: Any, model_file: Any) -> Any:  # type: ignore[override]
        _SUCCESS = object()

        def _wrap_result(value: Any) -> Any:
            # ``SentencePieceProcessor.LoadFromFile`` returns ``None`` on success.
            # We propagate a sentinel upwards so recursive callers can
            # differentiate between "handled" and "keep searching" without
            # altering the public return contract (which should remain ``None``).
            return _SUCCESS if value is None else value

        def _load_from_bytes(data: bytes | bytearray | memoryview) -> Any:
            try:
                return _wrap_result(self.LoadFromSerializedProto(bytes(data)))
            except Exception:  # pragma: no cover - defensive path
                return None

        preferred_mapping_keys = (
            "model",
            "path",
            "sentencepiece",
            "sentencepiece_model",
            "spm",
            "file",
        )

        visited: set[int] = set()

        def _attempt_load(candidate: Any) -> Any:
            candidate_id = id(candidate)
            if candidate_id in visited:
                return None
            visited.add(candidate_id)

            if candidate is None:
                return None

            if isinstance(candidate, str):
                try:
                    return _wrap_result(original_load(self, candidate))
                except Exception:
                    return None

            if isinstance(candidate, os.PathLike):
                try:
                    return _wrap_result(original_load(self, os.fspath(candidate)))
                except Exception:
                    return None

            fspath = getattr(candidate, "__fspath__", None)
            if callable(fspath):
                try:
                    resolved = fspath()
                except Exception:
                    resolved = None
                if isinstance(resolved, (str, bytes, bytearray, os.PathLike)):
                    return _attempt_load(resolved)

            if isinstance(candidate, (bytes, bytearray, memoryview)):
                return _load_from_bytes(candidate)

            if hasattr(candidate, "read") and callable(candidate.read):
                try:
                    data = candidate.read()
                except Exception:
                    data = None
                if isinstance(data, (bytes, bytearray, memoryview)):
                    return _load_from_bytes(data)
                return None

            if isinstance(candidate, MappingABC):
                for key in preferred_mapping_keys:
                    if key not in candidate:
                        continue
                    result = _attempt_load(candidate[key])
                    if result is not None:
                        return result
                for value in candidate.values():
                    result = _attempt_load(value)
                    if result is not None:
                        return result
                return None

            if isinstance(candidate, SequenceABC) and not isinstance(
                candidate,
                (bytes, bytearray, memoryview, str),
            ):
                for item in candidate:
                    result = _attempt_load(item)
                    if result is not None:
                        return result
                return None

            if hasattr(candidate, "__bytes__"):
                try:
                    data = bytes(candidate)
                except Exception:
                    pass
                else:
                    if isinstance(data, (bytes, bytearray)):
                        return _load_from_bytes(data)

            try:
                as_str = str(candidate)
            except Exception:
                as_str = None

            if isinstance(as_str, str) and as_str:
                try:
                    return _wrap_result(original_load(self, as_str))
                except TypeError:
                    pass
                except Exception:
                    raise

            return None

        result = _attempt_load(model_file)
        if result is _SUCCESS:
            return None
        if result is not None:
            return result

        raise TypeError("not a string")

    processor_cls.LoadFromFile = patched_load  # type: ignore[assignment]
    try:
        setattr(spm, "_mistral_sentencepiece_patch_applied", True)
    except Exception:
        pass


def _patch_transformers_tokenizers() -> None:
    try:
        from transformers.models.mistral3 import configuration_mistral3  # type: ignore
    except Exception:
        return

    config_cls = getattr(configuration_mistral3, "Mistral3Config", None)
    if config_cls is None:
        return

    model_type = getattr(config_cls, "model_type", None)
    if not isinstance(model_type, str) or not model_type:
        model_type = "mistral3"

    slow_cls, slow_name = _import_first_available(_TOKENIZER_SLOW_CANDIDATES)
    fast_cls, fast_name = _import_first_available(_TOKENIZER_FAST_CANDIDATES)

    if slow_cls is None and fast_cls is None:
        return

    try:
        from transformers.models.auto import auto_factory as tf_auto_factory  # type: ignore
        from transformers.models.auto import configuration_auto, tokenization_auto  # type: ignore
    except Exception:
        tf_auto_factory = configuration_auto = tokenization_auto = None

    applied = False

    if tokenization_auto is not None:
        mapping_names = getattr(tokenization_auto, "TOKENIZER_MAPPING_NAMES", None)
        if isinstance(mapping_names, Mapping):
            existing = mapping_names.get(model_type)
            if existing is not None:
                applied = True
            if existing is None or existing == (None, None):
                mapping_names[model_type] = (slow_name, fast_name)
                applied = True
            else:
                slow_entry, fast_entry = existing
                if slow_entry is None and slow_name is not None:
                    slow_entry = slow_name
                    applied = True
                if fast_entry is None and fast_name is not None:
                    fast_entry = fast_name
                    applied = True
                mapping_names[model_type] = (slow_entry, fast_entry)

        mapping = getattr(tokenization_auto, "TOKENIZER_MAPPING", None)
        if mapping is not None:
            try:
                mapping[config_cls]
                applied = True
            except KeyError:
                try:
                    register = getattr(mapping, "register", None)
                    if callable(register):
                        register(config_cls, slow_cls, fast_cls)
                        applied = True
                except Exception:
                    pass

        if (
            tf_auto_factory is not None
            and configuration_auto is not None
            and hasattr(tf_auto_factory, "_LazyAutoMapping")
        ):
            LazyMapping = getattr(tf_auto_factory, "_LazyAutoMapping")
            try:
                tokenization_auto.TOKENIZER_MAPPING = LazyMapping(
                    configuration_auto.CONFIG_MAPPING_NAMES,
                    tokenization_auto.TOKENIZER_MAPPING_NAMES,
                )
                applied = True
            except Exception:
                pass

        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception:
            AutoTokenizer = None

        if AutoTokenizer is not None:
            try:
                mapping_obj = getattr(tokenization_auto, "TOKENIZER_MAPPING", None)
                if mapping_obj is not None:
                    AutoTokenizer._tokenizer_mapping = mapping_obj  # type: ignore[attr-defined]
                    applied = True
            except Exception:
                pass

    if applied:
        return

    try:
        from transformers import AutoConfig, AutoTokenizer  # type: ignore
    except Exception:
        return

    if getattr(AutoTokenizer, "_mistral3_tokenizer_patch_applied", False):
        return

    original_from_pretrained = AutoTokenizer.from_pretrained.__func__  # type: ignore[attr-defined]

    @classmethod
    def patched_from_pretrained(
        cls,
        pretrained_model_name_or_path: Any,
        *init_inputs: Any,
        **kwargs: Any,
    ) -> Any:
        original_kwargs = dict(kwargs)
        try:
            return original_from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **dict(kwargs))
        except KeyError as exc:
            kwargs = original_kwargs

            config = kwargs.get("config")
            config_type = type(config).__name__ if config is not None else ""

            if config is None:
                config_kwargs: dict[str, Any] = {}
                for key in (
                    "cache_dir",
                    "force_download",
                    "local_files_only",
                    "proxies",
                    "revision",
                    "token",
                    "token_authentication",
                    "trust_remote_code",
                    "resume_download",
                    "subfolder",
                ):
                    if key in kwargs:
                        config_kwargs[key] = kwargs[key]
                try:
                    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)
                    config_type = type(config).__name__
                except Exception:
                    config = None
                    config_type = ""

            if "Mistral3Config" not in repr(exc) and config_type != "Mistral3Config":
                raise

            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("config", None)
            fallback_kwargs.pop("tokenizer_type", None)
            use_fast_raw = fallback_kwargs.pop("use_fast", True)
            use_fast = True if use_fast_raw is None else bool(use_fast_raw)

            order: list[type] = []
            if use_fast and fast_cls is not None:
                order.append(fast_cls)
            if slow_cls is not None and slow_cls not in order:
                order.append(slow_cls)
            if not use_fast and fast_cls is not None and fast_cls not in order:
                order.append(fast_cls)

            last_error: Exception | None = None
            for tokenizer_cls in order:
                try:
                    return tokenizer_cls.from_pretrained(
                        pretrained_model_name_or_path, *init_inputs, **fallback_kwargs
                    )
                except TypeError:
                    sanitised_kwargs = dict(fallback_kwargs)
                    sanitised_kwargs.pop("use_fast", None)
                    try:
                        return tokenizer_cls.from_pretrained(
                            pretrained_model_name_or_path, *init_inputs, **sanitised_kwargs
                        )
                    except Exception as inner_exc:  # pragma: no cover - defensive
                        last_error = inner_exc
                        continue
                except Exception as inner_exc:  # pragma: no cover - defensive
                    last_error = inner_exc
                    continue

            if last_error is not None:
                raise last_error
            raise

    AutoTokenizer.from_pretrained = patched_from_pretrained  # type: ignore[assignment]
    AutoTokenizer._mistral3_tokenizer_patch_applied = True


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

    def _describe_model_config(instance: Any) -> str:
        for attribute in ("name", "model", "path", "local_path"):
            if hasattr(instance, attribute):
                try:
                    value = getattr(instance, attribute)
                except Exception:
                    continue
                if value:
                    if isinstance(value, str):
                        normalised = _normalise_override_key(value)
                        if normalised is not None:
                            value = normalised
                    return f"{attribute}={value!r}"
        return f"ModelConfig(id={id(instance)})"

    def patched_layer_count(self: Any) -> int:
        config_obj = getattr(self, "config", None)
        preferred = None
        if config_obj is not None:
            preferred = _ensure_layer_attribute(config_obj)

        layer_range = getattr(self, "layer_range", None)
        range_count = _layer_range_length(layer_range, preferred=preferred)
        override_hint = None
        if config_obj is not None:
            for attr_name in ("_name_or_path", "name_or_path"):
                candidate = getattr(config_obj, attr_name, None)
                if isinstance(candidate, str):
                    normalised = _normalise_override_key(candidate)
                    if normalised is not None and normalised in _LAYER_OVERRIDE_MAP:
                        override_hint = _LAYER_OVERRIDE_MAP[normalised]
                        break
        if range_count is not None:
            if config_obj is not None:
                _set_config_attr(config_obj, _LAYER_OVERRIDE_ATTR, range_count)
                _set_config_attr(config_obj, "num_hidden_layers", range_count)
            final = range_count
        elif preferred is not None:
            final = preferred
        else:
            final = original_layer_count(self)

        _log_layer_debug(
            f"{_describe_model_config(self)} layer_range={layer_range!r} "
            f"preferred={preferred} range_count={range_count} override_hint={override_hint} final={final}"
        )
        return final

    ModelConfig.layer_count = patched_layer_count
    ModelConfig._mistral3_patch_applied = True


def _main() -> None:
    _patch_sentencepiece_loader()
    _patch_transformers_tokenizers()
    _patch_mergekit()


_main()
