print(f'Loading {__file__}...')

import skimage.io as io
import numpy as np
import time as ttime
import inspect

try:
    # cytools is a drop-in replacement for toolz, implemented in Cython
    from cytools import partition
except ImportError:
    from toolz import partition

from bluesky.plans import (
    count,
    list_scan,
    _check_detectors_type_input
)

from bluesky import plan_patterns, utils
from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from bluesky.protocols import Flyable, Movable, NamedMovable, Readable
from bluesky.utils import (
    CustomPlanMetadata,
    Msg,
    MsgGenerator,
    ScalarOrIterableFloat,
    get_hinted_fields,
)

# Newer scans


def setup_xrd_dets(dets,
                   dwell,
                   N_images):
    # Convenience function for setting up xrd detectors

    dets_by_name = {d.name : d for d in dets}

    # Setup merlin
    if 'merlin' in dets_by_name:
        xrd = dets_by_name['merlin']
        # Make sure we respect whatever the exposure time is set to
        if (dwell < 0.0066392):
            print('The Merlin should not operate faster than 7 ms.')
            print('Changing the scan dwell time to 7 ms.')
            dwell = 0.007
        # According to Ken's comments in hxntools, this is a de-bounce time
        # when in external trigger mode
        xrd.cam.stage_sigs['acquire_time'] = 0.75 * dwell  # - 0.0016392
        xrd.cam.stage_sigs['acquire_period'] = 0.75 * dwell + 0.0016392
        xrd.cam.stage_sigs['num_images'] = 1
        xrd.stage_sigs['total_points'] = N_images
        xrd.hdf5.stage_sigs['num_capture'] = N_images
        del xrd

    # Setup dexela
    if 'dexela' in dets_by_name:
        xrd = dets_by_name['dexela']
        xrd.stage_sigs['total_points'] = N_images
        xrd.cam.stage_sigs['acquire_time'] = dwell
        xrd.cam.stage_sigs['acquire_period'] = dwell
        xrd.cam.stage_sigs['num_images'] = N_images
        xrd.hdf5.stage_sigs['num_capture'] = N_images
        del xrd


# Assumes detector stage sigs are already set
# Treat like a plan stub and use within a run decorator
def _continuous_dark_fields(dets,
                            N_dark=10,
                            shutter=True):

    # Disable if no dark frames are requested
    if N_dark <= 0:
        return             

    dets_by_name = {d.name : d for d in dets}
    xrd_dets = []
    reset_sigs = []

    # # Merlin may not be needed...
    # if 'merlin' in dets_by_name:
    #     xrd = dets_by_name['merlin']
    #     sigs = OrderedDict(
    #         [   
    #         (xrd.cam, 'trigger_mode', 0),
    #         (xrd, 'total_points', N_dark),
    #         (xrd.cam, 'num_images', N_dark),
    #         (xrd.hdf5, 'num_capture', N_dark)
    #         ]
    #     )

    #     original_sigs = []
    #     for obj, key, value in sigs:
    #         if key in obj.stage_sigs:
    #             original_sigs.append((obj, key, obj.stage_sigs[key]))
    #         obj.stage_sigs[key] = value
        
    #     xrd_dets.append(xrd)
    #     reset_sigs.extend(original_sigs)

    if 'dexela' in dets_by_name:
        xrd = dets_by_name['dexela']
        sigs = [
                (xrd, 'total_points', N_dark),
                (xrd, 'cam.image_mode', 'Multiple'),
                (xrd.cam, 'trigger_mode', 'Int. Fixed Rate'),
                (xrd.cam, 'image_mode', 'Multiple'),
                (xrd.cam, 'num_images', N_dark),
                (xrd.hdf5, 'num_capture', N_dark),
                ]

        original_sigs = []
        for obj, key, value in sigs:
            if key in obj.stage_sigs:
                original_sigs.append((obj, key, obj.stage_sigs[key]))
            obj.stage_sigs[key] = value
        
        xrd_dets.append(xrd)
        reset_sigs.extend(original_sigs)
    
    if len(xrd_dets) > 0:
        d_status = shut_d.read()['shut_d_request_open']['value'] == 1 # is open
        if shutter: # Avoid printing banner
            yield from check_shutters(shutter, 'Close')
        print('Acquiring dark-field...')
        
        staging_list = [det._staged == Staged.yes for det in xrd_dets]
        mode_list = [det._mode for det in xrd_dets]
        for staged, det in zip(staging_list, xrd_dets):
            det._mode = SRXMode.fly # Not yield from???
            if staged:
                yield from bps.unstage(det)
            
            yield from bps.stage(det)
            # Change hard-coded stage values
            # Hacky implementation!
            yield from abs_set(det.cam.num_images, N_dark) # Swapped back
            # yield from abs_set(det.hdf5.num_capture, det.total_points.get())
            yield from abs_set(det.cam.trigger_mode, 'Int. Fixed Rate') # Swapped back from forced fly mode
        
        # Take images
        yield from bps.trigger_and_read(xrd_dets, name='dark')

        # Reset to original stage_sigs    
        for obj, key, value in reset_sigs:
            obj.stage_sigs[key] = value   
        
        for staged, mode, det in zip(staging_list, mode_list, xrd_dets):
            det._mode = mode # Not yield from???
            yield from bps.unstage(det)
            if staged:
                yield from bps.stage(det)

        # Clear descripter cache
        for det in xrd_dets:
            yield Msg("clear_describe_cache", det)

        if d_status and shutter: # Avoid printing banner
            yield from check_shutters(shutter, 'Open')


# Decorator version of dark fields. Must happen within open run.
def dark_decorator(dets, N_dark=10, shutter=True):
    def inner_decorator(func):
        @functools.wraps(func)
        def func_with_dark(*args, **kwargs):
            yield from _continuous_dark_fields(dets,
                                               N_dark=N_dark,
                                               shutter=shutter)
            yield from func(*args, **kwargs)
        return func_with_dark
    return inner_decorator


def energy_rocking_curve(e_low,
                         e_high,
                         e_num,
                         dwell,
                         xrd_dets,
                         N_dark=0,
                         vlm_snapshot=False,
                         snapshot_after=False,
                         shutter=True,
                         peakup_flag=True,
                         plotme=False,
                         return_to_start=True):

    start_energy = energy.energy.position

    # Convert to keV
    if e_low > 1000:
        e_low /= 1000
    if e_high > 1000:
        e_high /= 1000

    # Define some useful variables
    e_cen = (e_high + e_low) / 2
    e_range = np.linspace(e_low, e_high, e_num)

    # Define detectors
    dets = [xs, sclr1] + xrd_dets
    setup_xrd_dets(dets, dwell, e_num)

    # Set xs and sclr1 values
    # Poorly implemented to protect other scans
    sigs = [
            (xs, 'external_trig', False),
            (xs, 'total_points', e_num),
            (get_me_the_cam(xs), 'acquire_time', dwell),
            (sclr1, 'preset_time', dwell)
            ]
    original_sigs = []
    xs.mode = SRXMode.step
    for obj, key, value in sigs:
        original_sigs.append((obj, key, getattr(obj, key).get()))
        yield from abs_set(getattr(obj, key), value)

    # Defining scan metadata
    md = {}
    get_stock_md(md)
    md['scan']['type'] = 'ENERGY_RC'
    md['scan']['scan_input'] = [e_low, e_high, e_num, dwell]
    md['scan']['dwell'] = dwell
    md['scan']['detectors'] = [d.name for d in dets]

    # Live Callbacks
    livecallbacks = [LiveTable(['energy_energy', 'dexela_stats2_total'])]
    
    if plotme:
        livecallbacks.append(LivePlot('dexela_stats2_total', x='energy_energy'))

    # Move to center energy and perform peakup
    if peakup_flag:  # Find optimal c2_fine position
        print('Performing center energy peakup.')
        yield from mov(energy, e_cen)
        yield from peakup(shutter=shutter)
    
    @run_decorator(md=md)
    @vlm_decorator(vlm_snapshot, after=snapshot_after)
    @dark_decorator(dets, N_dark=N_dark, shutter=shutter) 
    def plan():
        yield from _continuous_dark_fields(dets, N_dark=N_dark, shutter=shutter)
        # Always check shutters to print banner
        yield from check_shutters(shutter, 'Open')
        yield from mod_list_scan(dets, energy, e_range, run_agnostic=True)
        if shutter: # Conditional check ot avoid banner
            yield from check_shutters(shutter, 'Close')
    
    # Plan with failed dark_field and default behavior
    # def plan():
    #     yield from check_shutter(shutter, 'Open')
    #     yield from list_scan(dets, energy, e_range, md=md)
    #     if shutter: # Conditional check ot avoid banner
    #         yield from check_shutters(shutter, 'Close')

    yield from subs_wrapper(plan(), {'all' : livecallbacks})

    # Reset xs and sclr1
    for obj, key, value in original_sigs:
        yield from abs_set(getattr(obj, key), value)
    
    if return_to_start:
        yield from mov(energy, start_energy)


def relative_energy_rocking_curve(e_range,
                                  e_num,
                                  dwell,
                                  xrd_dets,
                                  peakup_flag=False, # rewrite default
                                  **kwargs):
    
    en_current = energy.energy.position

    # Convert to keV. Not as straightforward as endpoint inputs
    if en_range > 5:
        warn_str = (f'WARNING: Assuming energy range of {en_range} '
                    + 'was given in eV.')
        print(warn_str)
        en_range /= 1000
    
    # Ensure energy.energy.positiion is reading correctly
    if en_current > 1000:
        en_current /= 1000

    e_low = en_current - (e_range / 2)
    e_high = en_current + (e_range / 2)
    
    yield from energy_rocking_curve(e_low,
                                    e_high,
                                    e_num,
                                    dwell,
                                    xrd_dets,
                                    peakup_flag=peakup_flag
                                    **kwargs)


def extended_energy_rocking_curve(e_low,
                                  e_high,
                                  e_num,
                                  dwell,
                                  xrd_dets,
                                  N_dark=0,
                                  shutter=True):

    # Breaking an extended energy rocking curve up into smaller pieces
    # The goal is to allow for multiple intermittent peakups

    # Convert to ev
    if e_low > 1000:
        e_low /= 1000
    if e_high > 1000:
        e_high /= 1000

    # Loose chunking at about 1000 eV
    e_range = e_high - e_low
    e_chunks = int(np.round(e_num / e_range))
    e_vals = np.linspace(e_low, e_high, e_num)

    e_rcs = [list(e_vals[i:i + e_chunks]) for i in range(0, len(e_vals), e_chunks)]
    e_rcs[-2].extend(e_rcs[-1])
    e_rcs.pop(-1)

    for e_rc in e_rcs:
        yield from energy_rocking_curve(e_rc[0],
                                        e_rc[-1],
                                        len(e_rc),
                                        dwell,
                                        xrd_dets,
                                        N_dark=N_dark,
                                        shutter=shutter,
                                        peakup_flag=True,
                                        plotme=False,
                                        return_to_start=False)


def angle_rocking_curve(th_low,
                        th_high,
                        th_num,
                        dwell,
                        xrd_dets,
                        N_dark=0,
                        vlm_snapshot=False,
                        snapshot_after=False,
                        shutter=True,
                        plotme=False,
                        return_to_start=True):
    
    # th in mdeg!!!

    start_th = nano_stage.th.user_readback.get()

    # Define some useful variables
    th_range = np.linspace(th_low, th_high, th_num)

    # Define detectors
    dets = [xs, sclr1] + xrd_dets
    setup_xrd_dets(dets, dwell, th_num)

    # Set xs and sclr1 values
    # Poorly implemented to protect other scans
    sigs = [
            (xs, 'external_trig', False),
            (xs, 'total_points', th_num),
            (get_me_the_cam(xs), 'acquire_time', dwell),
            (sclr1, 'preset_time', dwell)
            ]
    original_sigs = []
    xs.mode = SRXMode.step
    for obj, key, value in sigs:
        original_sigs.append((obj, key, getattr(obj, key).get()))
        yield from abs_set(getattr(obj, key), value)

    # Defining scan metadata
    md = {}
    get_stock_md(md)
    md['scan']['type'] = 'ANGLE_RC'
    md['scan']['scan_input'] = [th_low, th_high, th_num, dwell]
    md['scan']['dwell'] = dwell
    md['scan']['detectors'] = [d.name for d in dets]

    # Live Callbacks
    livecallbacks = [LiveTable(['nano_stage_th_user_setpoint', 'dexela_stats2_total'])]
    
    if plotme:
        livecallbacks.append(LivePlot('dexela_stats2_total', x='nano_stage_th_user_setpoint'))

    @run_decorator(md=md)
    @vlm_decorator(vlm_snapshot, after=snapshot_after)
    @dark_decorator(dets, N_dark=N_dark, shutter=shutter) 
    def plan():
        # Always check shutters to print banner
        yield from check_shutters(shutter, 'Open')
        yield from mod_list_scan(dets, nano_stage.th, th_range, run_agnostic=True)
        if shutter: # Conditional check ot avoid banner
            yield from check_shutters(shutter, 'Close')
    
    # Plan with failed dark_field and default behavior
    # def plan():
    #     yield from check_shutter(shutter, 'Open')
    #     yield from list_scan(dets, nano_stage.th, th_range, md=md)
    #     if shutter: # Conditional check ot avoid banner
    #         yield from check_shutters(shutter, 'Close')

    yield from subs_wrapper(plan(), {'all' : livecallbacks})

    # Reset xs and sclr1
    for obj, key, value in original_sigs:
        yield from abs_set(getattr(obj, key), value)

    if return_to_start:
        yield from mov(nano_stage.th, start_th)


def relative_angle_rocking_curve(th_range,
                                 th_num,
                                 dwell,
                                 xrd_dets,
                                 **kwargs):
    
    th_current = nano_stage.th.user_readback.get()
    th_low = th_current - (th_range / 2)
    th_high = th_current + (th_range / 2)

    yield from angle_rocking_curve(th_low,
                                   th_high,
                                   th_num,
                                   dwell,
                                   xrd_dets,
                                   **kwargs)


def flying_angle_rocking_curve(th_low,
                               th_high,
                               th_num,
                               dwell,
                               xrd_dets,
                               return_to_start=True,
                               **kwargs):
    # More direct convenience wrapper for scan_and_fly

    start_th = nano_stage.th.user_readback.get()
    y_current = nano_stage.y.user_readback.get()

    kwargs.setdefault('xmotor', nano_stage.th)
    kwargs.setdefault('ymotor', nano_stage.y)
    kwargs.setdefault('flying_zebra', nano_flying_zebra_coarse)
    yield from abs_set(kwargs['flying_zebra'].fast_axis, 'NANOHOR')
    yield from abs_set(kwargs['flying_zebra'].slow_axis, 'NANOVER')

    dets = [xs, sclr1] + xrd_dets

    yield from scan_and_fly_base(dets,
                                 th_low,
                                 th_high,
                                 th_num,
                                 y_current,
                                 y_current,
                                 1,
                                 dwell,
                                 **kwargs)
    
    # Is this needed for scan_and_fly_base???
    if return_to_start:
        yield from mov(nano_stage.th, start_th)

    
def relative_flying_angle_rocking_curve(th_range,
                                        th_num,
                                        dwell,
                                        xrd_dets,
                                        **kwargs):
    
    th_current = nano_stage.th.user_readback.get()
    th_low = th_current - (th_range / 2)
    th_high = th_current + (th_range / 2)

    yield from flying_angle_rocking_curve(th_low,
                                          th_high,
                                          th_num,
                                          dwell,
                                          xrd_dets,
                                          **kwargs)


# A static xrd measurement without changing energy or moving stages
def static_xrd(num,
               dwell,
               xrd_dets,
               N_dark=0,
               vlm_snapshot=False,
               snapshot_after=False,
               shutter=True,
               plotme=False):

    # Define detectors
    dets = [xs, sclr1] + xrd_dets
    setup_xrd_dets(dets, dwell, num)

    # Set xs and sclr1 values
    # Poorly implemented to protect other scans
    sigs = [
            (xs, 'external_trig', False),
            (xs, 'total_points', num),
            (get_me_the_cam(xs), 'acquire_time', dwell),
            (sclr1, 'preset_time', dwell)
            ]
    original_sigs = []
    xs.mode = SRXMode.step
    for obj, key, value in sigs:
        original_sigs.append((obj, key, getattr(obj, key).get()))
        yield from abs_set(getattr(obj, key), value)

    # Defining scan metadata
    md = {}
    get_stock_md(md)
    md['scan']['type'] = 'STATIC_XRD'
    md['scan']['scan_input'] = [num, dwell]
    md['scan']['dwell'] = dwell
    md['scan']['detectors'] = [sclr1.name] + [d.name for d in xrd_dets]
    md['scan']['energy'] = f'{energy.energy.position:.5f}'                                 
    md['scan']['start_time'] = ttime.ctime(ttime.time())

    # Live Callbacks
    livecallbacks = [LiveTable(['dexela_stats2_total'])]
    
    if plotme:
        livecallbacks.append(LivePlot('dexela_stats2_total'))

    @run_decorator(md=md)
    @vlm_decorator(vlm_snapshot, after=snapshot_after)
    @dark_decorator(dets, N_dark=N_dark, shutter=shutter)  
    def plan():
        # Always check shutters to print banner
        yield from check_shutters(shutter, 'Open')
        yield from mod_count(dets, num, run_agnostic=True)
        if shutter: # Conditional check to avoid banner
            yield from check_shutters(shutter, 'Close')
    
    # Plan with failed dark_field and default behavior
    # def plan():
    #     yield from check_shutter(shutter, 'Open')
    #     yield from count(dets, num, md=md)
    #     if shutter: # Conditional check to avoid banner
    #         yield from check_shutters(shutter, 'Close')

    yield from subs_wrapper(plan(), {'all' : livecallbacks})

    # Reset xs and sclr1
    for obj, key, value in original_sigs:
        yield from abs_set(getattr(obj, key), value)


# Run-agnostic standard scans

def agnostic_run_decorator(run_agnostic):
    def decorator(func):
        if run_agnostic:
            return func
        else:
            return run_decorator(func)
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
    if md is None:
        md = get_stock_md({})
    _md.update(md or {})

    # per_shot might define a different stream, so do not predeclare primary
    predeclare = per_shot is None and os.environ.get("BLUESKY_PREDECLARE", False)
    msg_per_step: PerShot = per_shot if per_shot else bps.one_shot

    @stage_decorator(detectors)
    @agnostic_run_decorator(run_agnostic)
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
    if md is None:
        md = get_stock_md({})
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
    if md is None:
        md = get_stock_md({})
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

    @stage_decorator(list(detectors) + motors)
    @agnostic_run_decorator(run_agnostic)
    def inner_scan_nd():
        if predeclare:
            yield from bps.declare_stream(*motors, *detectors, name="primary")
        for step in list(cycler):
            yield from per_step(detectors, step, pos_cache)

    return (yield from inner_scan_nd())