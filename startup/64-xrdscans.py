print(f'Loading {__file__}...')

import skimage.io as io
import numpy as np
import time as ttime

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
    # md['scan']['detectors'] = [d.name for d in dets]
    md_dets = dets
    if vlm_snapshot:
        md_dets = md_dets + [nano_vlm]
    get_det_md(md, md_dets)

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

    # Convert to kev
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


# Re-write to encompass a single scan ID with intelligent vlm and dark-field support
# TODO: livecallbacks may fail switching back and forth...
def continuous_energy_rocking_curve(e_low,
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

    # Convert to kev
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
    # md['scan']['detectors'] = [d.name for d in dets]
    md_dets = dets
    if vlm_snapshot:
        md_dets = md_dets + [nano_vlm]
    get_det_md(md, md_dets)

    # Live Callbacks
    livecallbacks = [LiveTable(['energy_energy', 'dexela_stats2_total'])]
    
    if plotme:
        livecallbacks.append(LivePlot('dexela_stats2_total', x='energy_energy'))

    @run_decorator(md=md)
    @vlm_decorator(vlm_snapshot, after=snapshot_after)
    @dark_decorator(dets, N_dark=N_dark, shutter=shutter)
    def plan():
        for iteration, e_rc in enumerate(e_rcs):
            # Move to center energy and perform peakup
            peakup_stream = f'00{iteration}_peakup'
            if peakup_flag:  # Find optimal c2_fine position
                print('Performing center energy peakup.')
                yield from mov(energy, e_rc[len(e_rc) // 2])
                yield from ra_smart_peakup(shutter=shutter,
                                           stream_name=peakup_stream)
            
            # Always check shutters to print banner
            if shutter or (not peakup_flag and i == 0): # Or first condition without peakup
                yield from check_shutters(shutter, 'Open')
            yield from mod_list_scan(dets, energy, e_rc, run_agnostic=True)
            if shutter: # Conditional check ot avoid banner
                yield from check_shutters(shutter, 'Close') 

    yield from subs_wrapper(plan(), {'all' : livecallbacks})

    # Reset xs and sclr1
    for obj, key, value in original_sigs:
        yield from abs_set(getattr(obj, key), value)
    
    if return_to_start:
        yield from mov(energy, start_energy)


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
    # md['scan']['detectors'] = [d.name for d in dets]
    md_dets = dets
    if vlm_snapshot:
        md_dets = md_dets + [nano_vlm]
    get_det_md(md, md_dets)

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
    md['scan']['start_time'] = ttime.ctime(ttime.time())
    # md['scan']['detectors'] = [d.name for d in dets]
    md_dets = dets
    if vlm_snapshot:
        md_dets = md_dets + [nano_vlm]
    get_det_md(md, md_dets)

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