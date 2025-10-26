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

from typing import Any, Iterable, Mapping, Sequence


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
        if alias:
            # Temporarily map the architecture name to the older handler
            # used for Mistral models.  We mutate the config in-place but
            # restore the original value afterwards to avoid side effects.
            replacement = (alias,)
            try:
                config.architectures = replacement
            except Exception:
                # Some configs expose architectures as a property without a
                # setter.  In that case we fall back to attribute tricks.
                # We still prefer to restore the original value afterwards.
                try:
                    object.__setattr__(config, "architectures", replacement)
                except Exception:
                    pass
            try:
                return original_get_architecture_info(config)
            finally:
                try:
                    config.architectures = architectures
                except Exception:
                    try:
                        object.__setattr__(config, "architectures", architectures)
                    except Exception:
                        pass
        return original_get_architecture_info(config)

    architecture.get_architecture_info = patched_get_architecture_info
    architecture._mistral3_patch_applied = True


def _main() -> None:
    _patch_mergekit()


_main()
