print(f"Loading {__file__}...")

import collections
import inspect
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from itertools import zip_longest
from typing import Any, Optional, Union

try:
    # cytools is a drop-in replacement for toolz, implemented in Cython
    from cytools import partition
except ImportError:
    from toolz import partition

from bluesky.plans import (
    _check_detectors_type_input
)

from bluesky import plan_patterns, utils
from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from bluesky.protocols import (
    Movable,
    Readable,
    Reading,
    Triggerable
)
from bluesky.utils import (
    CustomPlanMetadata,
    MsgGenerator,
    ScalarOrIterableFloat,
    plan
)

from bluesky.plans import (
    PerShot,
    PerStep
)



# Run-agnostic standard scans

def agnostic_run_decorator(run_agnostic, md=None):
    def decorator(func):
        if run_agnostic:
            return func
        else:
            return bpp.run_decorator(func, md=md)
    return decorator


# Re-write of bluesky count plan without run decorator
def mod_count(
            detectors: Sequence[Readable],
            num: Optional[int] = 1,
            delay: ScalarOrIterableFloat = 0.0,
            *,
            per_shot: Optional[PerShot] = None,
            md: Optional[CustomPlanMetadata] = None,
            run_agnostic=False,
            ) -> MsgGenerator[str]:
    """
    Take one or more readings from detectors.

    Parameters
    ----------
    detectors : list or tuple
        list of 'readable' objects
    num : integer, optional
        number of readings to take; default is 1

        If None, capture data until canceled
    delay : iterable or scalar, optional
        Time delay in seconds between successive readings; default is 0.
    per_shot : callable, optional
        hook for customizing action of inner loop (messages per step)
        Expected signature ::

           def f(detectors: Iterable[OphydObj]) -> Generator[Msg]:
               ...

    md : dict, optional
        metadata
    run_agnostic : bool, optional
        If True, a new run will not be created.

    Notes
    -----
    If ``delay`` is an iterable, it must have at least ``num - 1`` entries or
    the plan will raise a ``ValueError`` during iteration.
    """
    _check_detectors_type_input(detectors)
    _md = {}
    if md is None and not run_agnostic:
        md = get_stock_md({})
        get_det_md(md, [detectors])
    _md.update(md or {})

    # per_shot might define a different stream, so do not predeclare primary
    predeclare = per_shot is None and os.environ.get("BLUESKY_PREDECLARE", False)
    msg_per_step: PerShot = per_shot if per_shot else bps.one_shot

    @bpp.stage_decorator(detectors)
    @agnostic_run_decorator(run_agnostic, md=md)
    def inner_count() -> MsgGenerator[str]:
        if predeclare:
            yield from bps.declare_stream(*detectors, name="primary")
        return (yield from bps.repeat(partial(msg_per_step, detectors), num=num, delay=delay))

    return (yield from inner_count())



def mod_list_scan(
                detectors: Sequence[Readable],
                *args: tuple[Union[Movable, Any], list[Any]],
                per_step: Optional[PerStep] = None,
                md: Optional[CustomPlanMetadata] = None,
                run_agnostic=False
                ) -> MsgGenerator[str]:
    """
    Scan over one or more variables in steps simultaneously (inner product).

    Parameters
    ----------
    detectors : list or tuple
        list of 'readable' objects
    *args :
        For one dimension, ``motor, [point1, point2, ....]``.
        In general:

        .. code-block:: python

            motor1, [point1, point2, ...],
            motor2, [point1, point2, ...],
            ...,
            motorN, [point1, point2, ...]

        Motors can be any 'settable' object (motor, temp controller, etc.)

    per_step : callable, optional
        hook for customizing action of inner loop (messages per step)
        Expected signature:
        ``f(detectors, motor, step) -> plan (a generator)``
    md : dict, optional
        metadata
    run_agnostic : bool, optional
        If True, a new run will not be created.

    See Also
    --------
    :func:`bluesky.plans.rel_list_scan`
    :func:`bluesky.plans.list_grid_scan`
    :func:`bluesky.plans.rel_list_grid_scan`
    """
    _check_detectors_type_input(detectors)
    if len(args) % 2 != 0:
        raise ValueError("The list of arguments must contain a list of points for each defined motor")

    _md = {}
    if md is None and not run_agnostic:
        md = get_stock_md({})
        get_det_md(md, [detectors])
    _md.update(md or {})

    # set some variables and check that all lists are the same length
    lengths = {}
    motors: list[Any] = []
    pos_lists = []
    length = None
    for motor, pos_list in partition(2, args):
        pos_list = list(pos_list)  # Ensure list (accepts any finite iterable).
        lengths[motor.name] = len(pos_list)
        if not length:
            length = len(pos_list)
        motors.append(motor)
        pos_lists.append(pos_list)
    length_check = all(elem == list(lengths.values())[0] for elem in list(lengths.values()))

    if not length_check:
        raise ValueError(
            f"The lengths of all lists in *args must be the same. However the lengths in args are : {lengths}"
        )

    full_cycler = plan_patterns.inner_list_product(args)

    return (yield from mod_scan_nd(detectors,
                                   full_cycler,
                                   per_step=per_step,
                                   md=_md,
                                   run_agnostic=run_agnostic))



def mod_scan_nd(
                detectors: Sequence[Readable],
                cycler: Cycler,
                *,
                per_step: Optional[PerStep] = None,
                md: Optional[CustomPlanMetadata] = None,
                run_agnostic=False,
                ) -> MsgGenerator[str]:
    """
    Scan over an arbitrary N-dimensional trajectory.

    Parameters
    ----------
    detectors : list or tuple
    cycler : Cycler
        cycler.Cycler object mapping movable interfaces to positions
    per_step : callable, optional
        hook for customizing action of inner loop (messages per step).
        See docstring of :func:`bluesky.plan_stubs.one_nd_step` (the default)
        for details.
    md : dict, optional
        metadata
    run_agnostic : bool, optional
        If True, a new run will not be created.

    See Also
    --------
    :func:`bluesky.plans.inner_product_scan`
    :func:`bluesky.plans.grid_scan`

    Examples
    --------
    >>> from cycler import cycler
    >>> cy = cycler(motor1, [1, 2, 3]) * cycler(motor2, [4, 5, 6])
    >>> scan_nd([sensor], cy)
    """
    _check_detectors_type_input(detectors)
    _md = {}
    if md is None and not run_agnostic:
        md = get_stock_md({})
        get_det_md(md, [detectors])
    _md.update(md or {})

    predeclare = per_step is None and os.environ.get("BLUESKY_PREDECLARE", False)
    if per_step is None:
        per_step = bps.one_nd_step
    else:
        # Ensure that the user-defined per-step has the expected signature.
        sig = inspect.signature(per_step)

        def _verify_1d_step(sig):
            if len(sig.parameters) < 3:
                return False
            for name, (p_name, p) in zip_longest(["detectors", "motor", "step"], sig.parameters.items()):
                # this is one of the first 3 positional arguements, check that the name matches
                if name is not None:
                    if name != p_name:
                        return False
                # if there are any extra arguments, check that they have a default
                else:
                    if p.kind is p.VAR_KEYWORD or p.kind is p.VAR_POSITIONAL:
                        continue
                    if p.default is p.empty:
                        return False

            return True

        def _verify_nd_step(sig):
            if len(sig.parameters) < 3:
                return False
            for name, (p_name, p) in zip_longest(["detectors", "step", "pos_cache"], sig.parameters.items()):
                # this is one of the first 3 positional arguements, check that the name matches
                if name is not None:
                    if name != p_name:
                        return False
                # if there are any extra arguments, check that they have a default
                else:
                    if p.kind is p.VAR_KEYWORD or p.kind is p.VAR_POSITIONAL:
                        continue
                    if p.default is p.empty:
                        return False

            return True

        if sig == inspect.signature(bps.one_nd_step):
            pass
        elif _verify_nd_step(sig):
            # check other signature for back-compatibility
            pass
        elif _verify_1d_step(sig):
            # Accept this signature for back-compat reasons (because
            # inner_product_scan was renamed scan).
            dims = len(list(cycler.keys))
            if dims != 1:
                raise TypeError(f"Signature of per_step assumes 1D trajectory but {dims} motors are specified.")
            (motor,) = cycler.keys
            user_per_step = per_step

            def adapter(detectors, step, pos_cache):
                # one_nd_step 'step' parameter is a dict; one_id_step 'step'
                # parameter is a value
                (step,) = step.values()
                return (yield from user_per_step(detectors, motor, step))

            per_step = adapter  # type: ignore
        else:
            raise TypeError(
                "per_step must be a callable with the signature \n "
                "<Signature (detectors, step, pos_cache)> or "
                "<Signature (detectors, motor, step)>. \n"
                f"per_step signature received: {sig}"
            )
    pos_cache: dict = defaultdict(lambda: None)  # where last position is stashed
    cycler = utils.merge_cycler(cycler)
    motors = list(cycler.keys)

    @bpp.stage_decorator(list(detectors) + motors)
    @agnostic_run_decorator(run_agnostic, md=md)
    def inner_scan_nd():
        if predeclare:
            yield from bps.declare_stream(*motors, *detectors, name="primary")
        for step in list(cycler):
            yield from per_step(detectors, step, pos_cache)

    return (yield from inner_scan_nd())


def mod_grid_scan(
                  detectors: Sequence[Readable],
                  *args,
                 snake_axes: Optional[Union[Iterable, bool]] = None,
                 per_step: Optional[PerStep] = None,
                 md: Optional[CustomPlanMetadata] = None,
                 run_agnostic=False,
                 ) -> MsgGenerator[str]:
    """
    Scan over a mesh; each motor is on an independent trajectory.

    Parameters
    ----------
    detectors: list or tuple
        list of 'readable' objects
    ``*args``
        patterned like (``motor1, start1, stop1, num1,``
                        ``motor2, start2, stop2, num2,``
                        ``motor3, start3, stop3, num3,`` ...
                        ``motorN, startN, stopN, numN``)

        The first motor is the "slowest", the outer loop. For all motors
        except the first motor, there is a "snake" argument: a boolean
        indicating whether to following snake-like, winding trajectory or a
        simple left-to-right trajectory.
    snake_axes: boolean or iterable, optional
        which axes should be snaked, either ``False`` (do not snake any axes),
        ``True`` (snake all axes) or a list of axes to snake. "Snaking" an axis
        is defined as following snake-like, winding trajectory instead of a
        simple left-to-right trajectory. The elements of the list are motors
        that are listed in `args`. The list must not contain the slowest
        (first) motor, since it can't be snaked.
    per_step: callable, optional
        hook for customizing action of inner loop (messages per step).
        See docstring of :func:`bluesky.plan_stubs.one_nd_step` (the default)
        for details.
    md: dict, optional
        metadata
    run_agnostic : bool, optional
        If True, a new run will not be created.

    See Also
    --------
    :func:`bluesky.plans.rel_grid_scan`
    :func:`bluesky.plans.inner_product_scan`
    :func:`bluesky.plans.scan_nd`
    """
    # Notes: (not to be included in the documentation)
    #   The deprecated function call with no 'snake_axes' argument and 'args'
    #         patterned like (``motor1, start1, stop1, num1,``
    #                         ``motor2, start2, stop2, num2, snake2,``
    #                         ``motor3, start3, stop3, num3, snake3,`` ...
    #                         ``motorN, startN, stopN, numN, snakeN``)
    #         The first motor is the "slowest", the outer loop. For all motors
    #         except the first motor, there is a "snake" argument: a boolean
    #         indicating whether to following snake-like, winding trajectory or a
    #         simple left-to-right trajectory.
    #   Ideally, deprecated and new argument lists should not be mixed.
    #   The function will still accept `args` in the old format even if `snake_axes` is
    #   supplied, but if `snake_axes` is not `None` (the default value), it overrides
    #   any values of `snakeX` in `args`.

    _check_detectors_type_input(detectors)
    args_pattern = plan_patterns.classify_outer_product_args_pattern(args)
    if (snake_axes is not None) and (args_pattern == plan_patterns.OuterProductArgsPattern.PATTERN_2):
        raise ValueError(
            "Mixing of deprecated and new API interface is not allowed: "
            "the parameter 'snake_axes' can not be used if snaking is "
            "set as part of 'args'"
        )

    # For consistency, set 'snake_axes' to False if new API call is detected
    if (snake_axes is None) and (args_pattern != plan_patterns.OuterProductArgsPattern.PATTERN_2):
        snake_axes = False

    chunk_args = list(plan_patterns.chunk_outer_product_args(args, args_pattern))
    # 'chunk_args' is a list of tuples of the form: (motor, start, stop, num, snake)
    # If the function is called using deprecated pattern for arguments, then
    # 'snake' may be set True for some motors, otherwise the 'snake' is always False.

    # The list of controlled motors
    motors = [_[0] for _ in chunk_args]

    # Check that the same motor is not listed multiple times. This indicates an error in the script.
    if len(set(motors)) != len(motors):
        raise ValueError(f"Some motors are listed multiple times in the argument list 'args': '{motors}'")

    if snake_axes is not None:

        def _set_snaking(chunk, value):
            """Returns the tuple `chunk` with modified 'snake' value"""
            _motor, _start, _stop, _num, _snake = chunk
            return _motor, _start, _stop, _num, value

        if isinstance(snake_axes, collections.abc.Iterable) and not isinstance(snake_axes, str):
            # Always convert to a tuple (in case a `snake_axes` is an iterator).
            snake_axes = tuple(snake_axes)

            # Check if the list of axes (motors) contains repeated entries.
            if len(set(snake_axes)) != len(snake_axes):
                raise ValueError(f"The list of axes 'snake_axes' contains repeated elements: '{snake_axes}'")

            # Check if the snaking is enabled for the slowest motor.
            if len(motors) and (motors[0] in snake_axes):
                raise ValueError(f"The list of axes 'snake_axes' contains the slowest motor: '{snake_axes}'")

            # Check that all motors in the chunk_args are controlled in the scan.
            #   It is very likely that the script running the plan has a bug.
            if any([_ not in motors for _ in snake_axes]):  # noqa: C419
                raise ValueError(
                    f"The list of axes 'snake_axes' contains motors "
                    f"that are not controlled during the scan: "
                    f"'{snake_axes}'"
                )

            # Enable snaking for the selected axes.
            #   If the argument `snake_axes` is specified (not None), then
            #   any `snakeX` values that could be specified in `args` are ignored.
            for n, chunk in enumerate(chunk_args):
                if n > 0:  # The slowest motor is never snaked
                    motor = chunk[0]
                    if motor in snake_axes:
                        chunk_args[n] = _set_snaking(chunk, True)
                    else:
                        chunk_args[n] = _set_snaking(chunk, False)

        elif snake_axes is True:  # 'snake_axes' has boolean value `True`
            # Set all 'snake' values except for the slowest motor
            chunk_args = [_set_snaking(_, True) if n > 0 else _ for n, _ in enumerate(chunk_args)]
        elif snake_axes is False:  # 'snake_axes' has boolean value `True`
            # Set all 'snake' values
            chunk_args = [_set_snaking(_, False) for _ in chunk_args]
        else:
            raise ValueError(
                f"Parameter 'snake_axes' is not iterable, boolean or None: "
                f"'{snake_axes}', type: {type(snake_axes)}"
            )

    # Prepare the argument list for the `outer_product` function
    args_modified = []
    for n, chunk in enumerate(chunk_args):
        if n == 0:
            args_modified.extend(chunk[:-1])
        else:
            args_modified.extend(chunk)
    full_cycler = plan_patterns.outer_product(args=args_modified)

    _md = {}
    if md is None and not run_agnostic:
        md = get_stock_md({})
        get_det_md(md, [detectors])
    _md.update(md or {})

    return (yield from mod_scan_nd(detectors,
                                   full_cycler,
                                   per_step=per_step,
                                   md=_md,
                                   run_agnostic=run_agnostic))


@plan
def mod_trigger_and_read(devices: Sequence[Readable],
                         name: str = "primary",
                         timeout: float = None) -> MsgGenerator[Mapping[str, Reading]]:
    """
    Trigger and read a list of detectors and bundle readings into one Event.

    Parameters
    ----------
    devices : list
        devices to trigger (if they have a trigger method) and then read
    name : string, optional
        event stream name, a convenient human-friendly identifier; default
        name is 'primary'
    timeout : float, optional
        Time to wait before triggering and WaitTimeoutError. Does not wait by
        default matching the unmodified behavior.

    Returns
    -------
    readings:
        dict of device name to recorded information

    Yields
    ------
    msg : Msg
        messages to 'trigger', 'wait' and 'read'
    """
    from bluesky.preprocessors import contingency_wrapper

    # If devices is empty, don't emit 'create'/'save' messages.
    if not devices:
        yield from bps.null()
    devices = bps.separate_devices(devices)  # remove redundant entries
    rewindable = bps.all_safe_rewind(devices)  # if devices can be re-triggered

    def inner_trigger_and_read():
        grp = bps._short_uid("trigger")
        no_wait = True
        for obj in devices:
            if isinstance(obj, Triggerable):
                no_wait = False
                yield from bps.trigger(obj, group=grp)
        # Skip 'wait' if none of the devices implemented a trigger method.
        if not no_wait:
            yield from bps.wait(group=grp, timeout=timeout)
        yield from bps.create(name)

        def read_plan():
            ret = {}  # collect and return readings to give plan access to them
            for obj in devices:
                reading = yield from bps.read(obj)
                if reading is not None:
                    ret.update(reading)
            return ret

        def standard_path():
            yield from bps.save()

        def exception_path(exp):
            yield from bps.drop()
            raise exp

        ret = yield from contingency_wrapper(read_plan(), except_plan=exception_path, else_plan=standard_path)
        return ret

    from bluesky.preprocessors import rewindable_wrapper

    return (yield from rewindable_wrapper(inner_trigger_and_read(), rewindable))