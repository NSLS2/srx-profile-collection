print(f'Loading {__file__}...')
import time as ttime
from itertools import product
import bluesky.plan_stubs as bps


def blank_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def optimize_scalers(dwell=0.5,
                     scalers=['im', 'i0'],
                     upper_target=2E6,
                     lower_target=50,
                     shutter=True,
                     md=None,
                     stream_name=None
                     ):
    """
    Optimize scaler preamps.

    Parameters
    ----------
    dwell : float, optional
        Integration time for scaler in seconds. Default is 1.
    scalers : list, optional
        List of scaler keys in {'i0', 'im', 'it'} to optimize.
        Default is to only optmize on 'i0'.
    upper_target : float or list, optional
        Upper scaler target value in counts per second. Default is 1E6.
    lower_target : float or list, optional
        Lower scaler target value in counts per second. Default is 1E1.
    shutter : bool, optional
        Flag to indicate whether to control shutter. This should almost
        never be False. Default is True.
    md : dict, optional
        Dictionary of additional metadata for scan.
    """

    # Hard-coded variables
    settle_time = 0.1

    # Append to run or generate new
    RUN_WRAPPER = False
    if stream_name is None:
        RUN_WRAPPER = True
        stream_name = 'primary'

    # Check inputs
    for scaler_name in scalers:
        supported_scalers = ['i0', 'im', 'it']
        if scaler_name not in supported_scalers:
            err_str = (f'Scaler name {scaler_name} is not in '
                       + f'supported scalers {supported_scalers}.')
            raise ValueError(err_str)
    
    # Assembly preamps and detectors
    preamps, channel_names = [], []
    if 'i0' in scalers:
        preamps.append(i0_preamp)
        channel_names.append('sclr_i0')
    if 'im' in scalers:
        preamps.append(im_preamp)
        channel_names.append('sclr_im')
    if 'it' in scalers:
        preamps.append(it_preamp)
        channel_names.append('sclr_it')
    
    if isinstance(upper_target, (int, float)):
        upper_targets = [upper_target] * len(preamps)
    elif len(upper_target) == len(preamps):
        upper_targets = upper_target
    else:
        err_str = ("'upper_target' must be value"
                   + " or iterable matching length of scalers.")
        raise ValueError(err_str)
    
    if isinstance(lower_target, (int, float)):
        lower_targets = [lower_target] * len(preamps)
    elif len(lower_target) == len(preamps):
        lower_targets = lower_target
    else:
        err_str = ("'lower_target' must be value"
                   + " or iterable matching length of scalers.")
        raise ValueError(err_str)
    
    for target in lower_targets:
        if target <= 0:
            raise ValueError('Upper target must be greater than zero.')
    for target in lower_targets:
        if target < 0:
            raise ValueError('Lower larget must be greater than or equal to zero.')
    
    # Combo parameters of num (multiplier) and units
    preamp_combo_nums = list(product(range(3), range(9)))[::-1]

    # Add metadata
    _md = {'detectors' : [sclr1.name],
           'motors': [preamp.name for preamp in preamps],
           'plan_args' : {
               'dwell' : dwell,
               'upper_target' : upper_targets,
               'lower_target' : lower_targets
           },
           'plan_name' : 'optimize_scalers'
           }
    _md = get_stock_md(_md)
    _md['scan']['type'] = 'OPTIMIZE_SCALERS'
    _md['scan']['detectors'] = [sclr1.name]
    _md.update(md or {})

    # Setup dwell stage_sigs
    sclr1.stage_sigs['preset_time'] = dwell

    # Visualization
    livecb = []
    livecb.append(LiveTable(channel_names))

    # Need to add LivePlot, or LiveTable
    # @bpp.stage_decorator([sclr1])
    # @bpp.run_decorator(md = _md)
    # @bpp.subs_decorator(livecb)
    def optimize_all_preamps():

        # Optimize sensitivity
        # Turn off offset correction
        for idx in range(len(preamps)):
            yield from bps.mv(preamps[idx].offset_on, 0)

        # Open shutters
        yield from check_shutters(shutter, 'Open')

        opt_sens = [False,] * len(preamps)
        for combo_ind, combo in enumerate(preamp_combo_nums):
            # Break loop when completely optimized
            if all(opt_sens):
                break
            
            # Move preamps to new values
            yield Msg('checkpoint')
            for idx in range(len(preamps)):
                if opt_sens[idx]:
                    continue
                yield from bps.mv(
                    preamps[idx].sens_num, combo[1],
                    preamps[idx].sens_unit, combo[0]
                )
            yield from bps.sleep(settle_time) # Settle electronics?
            yield Msg('create', None, name=stream_name)
            yield Msg('trigger', sclr1, group='B')
            yield Msg('wait', None, 'B')

            # Read and iterate though all channels of interest
            ch_vals = yield Msg('read', sclr1)
            for idx in range(len(preamps)):
                if opt_sens[idx] or combo_ind == 0:
                    continue
                
                # Check if values have surpassed target value
                val = ch_vals[channel_names[idx]]['value']
                if val / dwell > upper_targets[idx]:
                    # print(f'{val} is greater than upper target for {channel_names[idx]}')
                    # print(f'{channel_names[idx]} parameters for exceeded values are {combo}')
                    # print(f'New parameters will be {preamp_combo_nums[combo_ind - 1]}')
                    # If true, dial back parameters and mark as optimized
                    yield from bps.mv(
                        preamps[idx].sens_num,
                        preamp_combo_nums[combo_ind - 1][1],
                        preamps[idx].sens_unit,
                        preamp_combo_nums[combo_ind - 1][0]
                    )
                    opt_sens[idx] = True
            yield Msg('save')

        # Optimize offsets
        # Close shutters
        yield from check_shutters(shutter, 'Close')
        print('extra wait')
        yield from bps.sleep(10 + settle_time)

        # Take baseline measurement without offsets
        yield Msg('checkpoint')
        yield Msg('create', None, name=stream_name)
        yield Msg('trigger', sclr1, group='B')
        # yield Msg('trigger', motor, group='B') # What does this one do???
        yield Msg('wait', None, 'B')

        direction_signs = []
        # Read and iterate though all channels of interest
        ch_vals = yield Msg('read', sclr1)
        for idx in range(len(preamps)):
            val = ch_vals[channel_names[idx]]['value']
            # Find and set offset sign
            if val > 10: # slightly more than zero
                direction_signs.append(1)
                yield from bps.mv(preamps[idx].offset_sign, 0)
                yield from bps.sleep(settle_time)
                # print(f'{channel_names[idx]} is positive')
            else:
                direction_signs.append(-1)
                yield from bps.mv(preamps[idx].offset_sign, 1)
                yield from bps.sleep(settle_time)

        yield Msg('save')

        # Turn offsets back on
        for idx in range(len(preamps)):
            yield from bps.mv(preamps[idx].offset_on, 1)
            yield from bps.sleep(settle_time)

        # Iterate through combinations
        opt_off = [False,] * len(preamps)
        for combo_ind, combo in enumerate(preamp_combo_nums):
            # Break loop when completely optimized
            if all(opt_off):
                break
            
            # Move preamps to new values
            yield Msg('checkpoint')
            for idx in range(len(preamps)):
                if opt_off[idx]:
                    continue
                yield from bps.mv(
                    preamps[idx].offset_num, combo[1],
                    preamps[idx].offset_unit, combo[0]
                )
            yield from bps.sleep(settle_time)
            yield Msg('create', None, name=stream_name)
            yield Msg('trigger', sclr1, group='B')
            # yield Msg('trigger', motor, group='B') # What does this one do???
            yield Msg('wait', None, 'B')

            # Read and iterate though all channels of interest
            ch_vals = yield Msg('read', sclr1)
            for idx in range(len(preamps)):
                # Read values
                val = ch_vals[channel_names[idx]]['value'] / dwell

                # Skip already optimized preamps
                if opt_off[idx]:
                    continue

                # Check if offset sign is correct from first reading
                elif combo_ind == 0:
                    if (val > 1 and direction_signs[idx] == 1):
                        # Flip offset sign, while off for safety
                        yield from bps.mv(preamps[idx].offset_sign, 1)
                        yield from bps.sleep(settle_time)
                        # print(f'Cond 1: Flipping offset sign for {preamps[idx].name}')

                    elif (val <= 1 and direction_signs[idx] == -1):
                        # Flip offset sign, while off for safety
                        yield from bps.mv(preamps[idx].offset_sign, 0)
                        yield from bps.sleep(settle_time)
                        # print(f'Cond 2 : Flipping offset sign for {preamps[idx].name}')          
                    continue
                
                # Check if offset is correct
                if (direction_signs[idx] > 0 and val >= lower_targets[idx]):
                    yield from bps.mv(
                        preamps[idx].offset_num,
                        preamp_combo_nums[combo_ind][1],
                        preamps[idx].offset_unit,
                        preamp_combo_nums[combo_ind][0]
                    )
                    opt_off[idx] = True
                elif (direction_signs[idx] < 0 and val <= lower_targets[idx]):
                    yield from bps.mv(
                        preamps[idx].offset_num,
                        preamp_combo_nums[combo_ind - 1][1],
                        preamps[idx].offset_unit,
                        preamp_combo_nums[combo_ind - 1][0]
                    )
                    opt_off[idx] = True

            yield Msg('save')
    
    # Open and close run, or append to other run
    if RUN_WRAPPER:
        @bpp.stage_decorator([sclr1])
        @bpp.run_decorator(md = _md)
        @bpp.subs_decorator(livecb)
        def plan():
            yield from optimize_all_preamps()
    else:
        @bpp.stage_decorator([sclr1])
        @bpp.subs_decorator(livecb)
        def plan():
            yield from optimize_all_preamps()
    
    # return (yield from optimize_all_preamps())
    # return (yield from plan())
    uid = (yield from plan())
    sclr1.stage_sigs.pop('preset_time', None)
    return uid


## WIP
"""
def align_diamond_aperture(dwell=0.1,
                           bin_low=934,
                           bin_high=954,
                           **kwargs):
    
    if bin_low is None or bin_high is None:
        if xs.channel01.mcaroi01.size_x.get() != 0:
            bin_low = xs.channel01.mcaroi01.min_x.get()
            bin_high = bin_low + xs.channel01.mcaroi01.size_x.get()
        else:
            raise ValueError('Must define bin_high and bin_low or set roi on Xpress3.')
    

    start_x = diamond_aperture.x.user_readback.get()
    start_y = diamond_aperture.y.user_readback.get()
    
    # Vertical alignment to fly in y
    yield from scan_and_fly_base(
                    [xs],
                    -12, 12, 241, start_x, start_x, 1, dwell,
                    xmotor = diamond_aperture.y,
                    ymotor = diamond_aperture.x
                    **kwargs
                )
"""


@parameter_annotation_decorator({
    "parameters": {
        "motor": {"default": "dcm.c2_fine"},
        "detectors": {"default": ['bpm3', 'bpm4', 'xbpm2']},
    }
})
def ra_smart_peakup(start=None,
                 min_step=0.005,
                 max_step=0.50,
                 *,
                 shutter=True,
                 motor=dcm.c2_fine,
                 detectors=[dcm.c2_pitch, bpm4, xbpm2],
                 target_fields=['bpm4_total_current', 'xbpm2_sumT'],
                 MAX_ITERS=100,
                 md=None,
                 stream_name=None,
                 verbose=False):
    """
    Quickly optimize X-ray flux into the SRX D-hutch based on
    measurements from two XBPMs.

    Parameters
    ----------
    start : float
        starting position of motor
    min_step : float
        smallest step for determining convergence
    max_step : float
        largest step for initial scanning
    motor : object
        any 'settable' object (motor, temp controller, etc.)
    detectors : list
        list of 'readable' objects
    target_fields : list
        list of strings with the data field for optimization
    MAX_ITERS : int, default=100
        maximum number of iterations for each target field
    md : dict, optional
        metadata
    verbose : boolean, optional
        print debugging information

    See Also
    --------
    :func:`bluesky.plans.adaptive_scan`
    """
    # Debugging print
    if verbose:
        print('Additional debugging is enabled.')

    # Append to run or generate new
    RUN_WRAPPER = False
    if stream_name is None:
        RUN_WRAPPER = True
        stream_name = 'primary'

    # Check min/max steps
    if not 0 < min_step < max_step:
        raise ValueError("min_step and max_step must meet condition of "
                         "max_step > min_step > 0")

    # Grab starting point
    if start is None:
        start = motor.readback.get()
        if verbose:
            print(f'Starting position: {start:.4}')

    # Check if bpm4 is working
    if 'bpm4' in [det.name for det in detectors]:
        # Need to implement
        # Or, hopefully, new device will not have this issue
        pass

    # Check foils
    if 'bpm4_total_current' in target_fields:
        E = energy.energy.readback.get()  # keV
        y = bpm4_pos.y.user_readback.get()  # Cu: y=0, Ti: y=25
        if np.abs(y-25) < 5:
            foil = 'Ti'
        elif np.abs(y) < 5:
            foil = 'Cu'
        else:
            foil = ''
            banner('Unknown foil! Continuing without check!')

        if verbose:
            print(f'Energy: {E:.4}')
            print(f'Foil:\n  {y=:.4}\n  {foil=}')

        threshold = 8.979
        if E > threshold and foil == 'Ti':
            banner('WARNING! BPM4 foil is not optimized for the incident energy.')
        elif E < threshold and foil == 'Cu':
            banner('WARNING! BPM4 foil is not optimized for the incident energy.')

    # We do not need the fast shutter open, so we will only check the B-shutter
    if shutter is True:
        if shut_b.status.get() == 'Not Open':
            print('Opening B-hutch shutter..')
            try:
                st = yield from mov(shut_b, "Open")
            # Need to figure out exception raises when shutter cannot open
            except Exception as ex:
                print(st)
                raise ex

    # Add metadata
    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor.name],
           'plan_args': {'detectors': list(map(repr, detectors)),
                         'motor': repr(motor),
                         'start': start,
                         'min_step': min_step,
                         'max_step': max_step,
                         },
           'plan_name': 'smart_peakup',
           'hints': {},
           }
    _md = get_stock_md(_md)
    _md['scan']['type'] = 'PEAKUP'
    _md['scan']['detectors'] = [det.name for det in detectors]
    _md.update(md or {})

    try:
        dimensions = [(motor.hints['fields'], stream_name)]
    except (AttributeError, KeyError):
        pass
    else:
        _md['hints'].setdefault('dimensions', dimensions)

    # Visualization
    livecb = []
    if verbose is False:
        livecb.append(LiveTable([motor.readback.name] + target_fields))

    # Need to add LivePlot, or LiveTable
    # @bpp.stage_decorator(list(detectors) + [motor])
    # @bpp.run_decorator(md=_md)
    # @bpp.subs_decorator(livecb)
    def smart_max_core(x0):
        # Optimize on a given detector
        def optimize_on_det(target_field, x0):
            past_pos = x0
            next_pos = x0
            step = max_step
            past_I = None
            cur_I = None
            cur_det = {}

            for N in range(MAX_ITERS):
                yield Msg('checkpoint')
                if verbose:
                    print(f'Moving {motor.name} to {next_pos:.4f}')
                yield from bps.mv(motor, next_pos)
                yield from bps.sleep(0.500)
                yield Msg('create', None, name=stream_name)
                for det in detectors:
                    yield Msg('trigger', det, group='B')
                yield Msg('trigger', motor, group='B')
                yield Msg('wait', None, 'B')
                for det in utils.separate_devices(detectors + [motor]):
                    cur_det = yield Msg('read', det)
                    if target_field in cur_det:
                        cur_I = cur_det[target_field]['value']
                        if verbose:
                            print(f'New measurement on {target_field}: {cur_I:.4}')
                yield Msg('save')

                # special case first first loop
                if past_I is None:
                    past_I = cur_I
                    next_pos += step
                    if verbose:
                        print(f'past_I is None. Continuing...')
                    continue

                dI = cur_I - past_I
                if verbose:
                    print(f'{dI=:.4f}')
                if dI < 0:
                    step = -0.6 * step
                else:
                    past_pos = next_pos
                    past_I = cur_I
                next_pos = past_pos + step
                if verbose:
                    print(f'{next_pos=:.4f}')

                # Maximum found
                if np.abs(step) < min_step:
                    if verbose:
                        print(f'Maximum found for {target_field} at {x0:.4f}!\n  {step=:.4f}')
                    return next_pos
            else:
                raise Exception('Optimization did not converge!')

        # Start optimizing based on each detector field
        for target_field in target_fields:
            if verbose:
                print(f'Optimizing on detector {target_field}')
            x0 = yield from optimize_on_det(target_field, x0)
    
    # Open and close run, or append to other run
    if RUN_WRAPPER:
        @bpp.stage_decorator(list(detectors) + [motor])
        @bpp.run_decorator(md=_md)
        @bpp.subs_decorator(livecb)
        def plan(start):
            yield from smart_max_core(start)
    else:
        @bpp.stage_decorator(list(detectors) + [motor])
        @bpp.subs_decorator(livecb)
        def plan(start):
            yield from smart_max_core(start)

    return (yield from plan(start))
