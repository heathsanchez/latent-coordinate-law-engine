"""Frozen, source-independent signature binding for V21 source-D trial.

Only two shapes are admitted: iterable first, or a known size knob followed
by a named iterable parameter. The candidate search never uses source names.
"""
import inspect
import itertools


SIZE_KNOBS = frozenset(('n', 'size', 'window_size', 'length', 'count'))
INPUT_NAMES = frozenset(('seq', 'sequence', 'iterable', 'xs', 'values', 'items', 'it'))
POSITIONAL = (inspect.Parameter.POSITIONAL_ONLY,
              inspect.Parameter.POSITIONAL_OR_KEYWORD)


def candidate_bindings(function):
    """Return bounded, JSON-serializable input/size binding variants."""
    try:
        params = list(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        return []
    if not params or params[0].kind not in POSITIONAL:
        return []

    if (params[0].name in SIZE_KNOBS and len(params) >= 2
            and params[1].kind in POSITIONAL
            and params[1].name in INPUT_NAMES):
        input_slot = 1
        remaining = params[2:]
        size_value = 2
    elif params[0].name not in SIZE_KNOBS:
        input_slot = 0
        remaining = params[1:]
        size_value = None
    else:
        return []

    knobs = {}
    for param in remaining:
        if param.name in SIZE_KNOBS and param.kind in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD):
            knobs[param.name] = 2
        elif param.default is inspect.Parameter.empty:
            return []

    return [dict(input_slot=input_slot, size_value=size_value, kwargs=knobs,
                 materialize=materialize) for materialize in (False, True)]


def call_bounded(function, variant, xs):
    """Invoke a declared binding; materialize no more than n observable items."""
    seq = list(xs)
    if variant['input_slot'] == 0:
        result = function(seq, **variant['kwargs'])
    elif variant['input_slot'] == 1 and variant['size_value'] == 2:
        result = function(2, seq, **variant['kwargs'])
    else:
        raise ValueError('UNAUTHORIZED_INPUT_BINDING')
    if variant['materialize']:
        result = list(itertools.islice(result, max(1, len(seq))))
    return result
