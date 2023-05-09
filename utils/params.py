import warnings
from argparse import Namespace
from typing import Dict, Optional, Union


def merge_params(
    base_params: Union[Namespace, Dict],
    override_params: Optional[Dict],
) -> Union[Namespace, Dict]:
    """
    Update the parameters in ``base_params`` with ``override_params``.
    Can be useful to override parsed command line arguments.'

    Args:
        base_params: Namespace or dictionary base parameters.
        override_params: Optional dictionary of parameters to override.
            Usually the parameters got from ``get_next_parameters()``.
            When it is none, nothing will happen.

    Returns:
        The updated ``base_params``. Note that ``base_params`` will
        be updated inplace. The return value is only for convenience.

    """
    if override_params is None:
        return base_params
    _is_dict = isinstance(base_params, dict)

    if isinstance(base_params, dict):
        _is_dict = True
    else:
        _is_dict = False
        base_params = vars(base_params)

    for __k, __v in override_params.items():
        if __k not in base_params:
            warnings.warn(
                f"The parameter {__k} is not in the base parameters. "
                f"Adding {__k} to the base parameters ..."
            )
        __t = type(base_params[__k])
        if type(__v) != __t and base_params[__k] is not None:
            warnings.warn(
                f"Expected {__k} in override parameters to have type {__t}, "
                f"but found type {type(__v)} of value {__v} instead. "
                f"Overriding anyway ..."
            )
        base_params[__k] = __v

    return base_params if _is_dict else Namespace(**base_params)


def print_params(params: Union[Namespace, Dict]):
    """Print the parameters with proper indentation.

    Args:
        params: Namespace or dictionary base parameters.

    """
    _params = params if isinstance(params, dict) else vars(params)
    _max_key_len = max([len(__k) for __k in _params.keys()])
    for __k, __v in _params.items():
        print(f"{__k:{_max_key_len + 4}s}: {__v}")
