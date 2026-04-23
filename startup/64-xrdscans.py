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
        xrd.cam.stage_sigs['acquire_time'] = 0.9*dwell - 0.002
        xrd.cam.stage_sigs['acquire_period'] = 0.9*dwell
        xrd.cam.stage_sigs['num_images'] = N_images
        xrd.stage_sigs['total_points'] = N_images
        xrd.hdf5.stage_sigs['num_capture'] = N_images
        del xrd

    # Setup dexela
    if 'dexela' in dets_by_name:
        xrd = dets_by_name['dexela']
        xrd.cam.acquire.set(0) # Halt any current acquisition
        xrd.stage_sigs['total_points'] = N_images
        xrd.cam.stage_sigs['acquire_time'] = dwell
        xrd.cam.stage_sigs['acquire_period'] = dwell
        xrd.cam.stage_sigs['num_images'] = N_images
        xrd.hdf5.stage_sigs['num_capture'] = N_images
        del xrd

    # Setup eiger
    if 'eiger' in dets_by_name:
        xrd = dets_by_name['eiger']
        xrd.cam.acquire.set(0)
        xrd.stage_sigs['total_points'] = N_images
        xrd.cam.stage_sigs['num_triggers'] = N_images
        xrd.hdf5.stage_sigs['num_capture'] = N_images

        # print('New Eiger stage sigs')
        # Sets bit-depth for fly-mode, otherwise actual time
        xrd.cam.stage_sigs['acquire_time'] = dwell - 0.010 # 10 ms is a lot, but dropping too many frames
        xrd.cam.stage_sigs['acquire_period'] = dwell

        # Update energy thresholds
        # Should do this for merlin too...
        xrd.cam.stage_sigs['photon_energy'] = 1e3 * np.round(energy.energy.setpoint.get())
        xrd.cam.stage_sigs['threshold_energy'] = 1e3 * 0.5 * np.round(energy.energy.setpoint.get())
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


# Base function for step-based static reciprocal space maps
@append_srx_kwargs_md
def step_rsm_base(start, stop, num,
                  dwell,
                  xrd_dets,
                  rocking_motor,
                  md=None,
                  N_dark=10,
                  vlm_snapshot=True,
                  shutter=True,
                  peakup_flag=True,
                  plotme=True,
                  return_to_start=True):

    if rocking_motor == energy:
        rocking_axis = 'ENERGY'
        curr_pos = energy.energy.setpoint.get()
        cen_val = (start + stop) / 2
    elif rocking_motor in [nano_stage.th, comp_th]:
        rocking_axis = 'ANGLE'
        curr_pos = nano_stage.th.user_setpoint.get()
        cen_val = None
    else:
        err_str = f'Rocking motor {rocking_motor} is not supported.'
        raise ValueError(err_str)

    # Get the points
    points = np.linspace(start, stop, num)
    
    # Setup detectors
    dets = [xs, slcr1] + xrd_dets
    setup_xrd_dets(dets, dwell, num)

    # Some pseudo stage-sigs to protect xs and sclr1
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
    md = get_stock_md(md)
    md['scan']['type'] = f'{rocking_axis}_RC'
    md['scan']['scan_input'] = [start, stop, num, dwell]
    md['scan']['dwell'] = dwell
    md_dets = dets
    if vlm_snapshot is True:
        md_dets = md_dets + [nano_vlm]
    get_det_md(md, md_dets)
    
    # Define some helper functions
    def at_scan(name, doc):
        time_rem = len(e_range) * (dwell + 4.25) # Some overhead for estimate
        scanrecord.time_remaining.put(time_rem / 3600)
        scanreocrd.time_rem_str.put(time_rem_convert(time_rem))

    def finalize_scan():
        yield from abs_set(scanrecord.scanning, False)
        yield from abs_set(scanrecord.time_remaining, 0)
        yield from abs_set(scanrecord.time_rem_str, time_rem_convert(0))

        # Reset xs and sclr1
        for obj, key, value in original_sigs:
            yield from abs_set(getattr(obj, key), value)
        
        if return_to_start:
            yield from mov(rocking_motor, curr_pos)

    def time_per_point(name, doc, st=ttime.time()):
        if (name == "event"):
            if ('seq_num' in doc.keys()):
                scanrecord.time_remaining.put((doc['time'] - st) / doc['seq_num'] *
                                                (len(ept) - doc['seq_num']) / 3600)
                scanrecord.time_rem_str.put(time_rem_convert(
                    ((doc['time'] - st) / doc['seq_num']) * # average time per point
                    (len(ept) - doc['seq_num']) # remaining number of points
                ))

    # Live Callbacks
    livecallbacks = [LiveTable(['energy_energy', 'dexela_stats2_total'])]
    livecallbacks.append(time_per_point)
    
    if plotme:
        livecallbacks.append(LivePlot('dexela_stats2_total', x='energy_energy'))

    # Move to center energy and perform peakup
    if peakup_flag and rocking_axis == 'ENERGY':
        print('Performing center energy peakup.')
        yield from mov(rocking_motor, cen_val)
        yield from peakup(shutter=shutter)

    # Define actual plan
    @run_decorator(md=md)
    @vlm_decorator(vlm_snapshot, after=True)
    @dark_decorator(dets, N_dark=N_dark, shutter=shutter) 
    def plan():
        # Always check shutters to print banner
        yield from check_shutters(shutter, 'Open')
        yield from mod_list_scan(dets, rocking_motor, points, run_agnostic=True)
        if shutter: # Conditional check ot avoid banner
            yield from check_shutters(shutter, 'Close')

    # Plan must be called to return the generators
    plan = finalize_wrapper(plan(), finalize_scan)
    # This actually runs the plan
    yield from subs_wrapper(plan, {'all' : livecallbacks,
                                   'start' : at_scan})


@append_srx_kwargs_md
def extended_energy_rsm(start, stop, num,
                        dwell,
                        xrd_dets,
                        md=None,
                        N_dark=10,
                        vlm_snapshot=True,
                        shutter=True,
                        chunk_range=1000, # in eV!
                        peakup_flag=True,
                        plotme=True,
                        return_to_start=True):

    # Breaking an extended energy rocking curve up into smaller pieces
    # The goal is to allow for multiple intermittent peakups

    # Convert to eV
    if start < 1e3 or stop < 1e3:
        start *= 1e3
        stop *= 1e3

    start_energy = energy.energy.setpoint.get()

    # Loose chunking at about 1000 eV
    pts_range = np.abs(stop - start)
    pts_chunks = int(np.round((num * chunk_range) / pts_range))
    pts_vals = np.linspace(start, stop, num)

    pts_rsms = [list(pts_vals[i:i + pts_chunks]) for i in range(0, len(pts_vals), pts_chunks)]
    if len(pts_rsms) > 1:
        pts_rsms[-2].extend(pts_rsms.pop(-1))

    # Parse one ore multiple scans
    if len(pts_rsms) != 1:
        peakup_flag = True
        plotme = False

    def plan():
        for i, pts_rsm in enumerate(pts_rsms):
            # No dark-field or vlm snapshots in intermediate scans
            if i != 0:
                N_dark = 0
                vlm_snapshot=False

            yield from step_rsm_base(pts_rsm[0], pts_rsm[-1], len(pts_rsm),
                                    dwell,
                                    xrd_dets,
                                    md=md,
                                    rocking_motor=energy,
                                    N_dark=N_dark,
                                    vlm_snapshot=vlm_snapshot,
                                    shutter=shutter,
                                    peakup_flag=peakup_flag,
                                    plotme=plotme,
                                    return_to_start=False)
            
    def finalize_scan():
        if return_to_start:
            yield from mov(energy, start_energy)

    # Wrap and run plan
    yield from finalize_wrapper(plan(), finalize_scan)
    

@append_srx_kwargs_md
def continuous_energy_rsm(start, stop, num,
                          dwell,
                          xrd_dets,
                          md=None,
                          N_dark=10,
                          vlm_snapshot=True,
                          shutter=True,
                          chunk_range=1000, # in eV!
                          peakup_flag=True,
                          plotme=True,
                          return_to_start=True):
    
    raise NotImplementedError()
    start_energy = energy.energy.readback.get()

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
    md = get_stock_md(md)
    md['scan']['type'] = 'ENERGY_RC'
    md['scan']['scan_input'] = [e_low, e_high, e_num, dwell]
    md['scan']['dwell'] = dwell
    # md['scan']['detectors'] = [d.name for d in dets]
    md_dets = dets
    if vlm_snapshot is True:
        md_dets = md_dets + [nano_vlm]
    get_det_md(md, md_dets)

    # Live Callbacks
    livecallbacks = [LiveTable(['energy_energy', 'dexela_stats2_total'])]
    
    if plotme:
        livecallbacks.append(LivePlot('dexela_stats2_total', x='energy_energy'))

    @run_decorator(md=md)
    @vlm_decorator(vlm_snapshot, after=True)
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


def flying_angle_rsm(start, stop, num,
                     dwell,
                     xrd_dets,
                     return_to_start=True,
                     **kwargs
                     ):

    # More direct convenience wrapper for scan_and_fly

    start_th = nano_stage.th.user_setpoint.get()
    y_current = nano_stage.y.user_setpoint.get()

    kwargs.setdefault('xmotor', nano_stage.th)
    kwargs.setdefault('ymotor', nano_stage.y)
    kwargs.setdefault('flying_zebra', nano_flying_zebra_coarse)
    yield from abs_set(kwargs['flying_zebra'].fast_axis, 'NANOHOR')
    yield from abs_set(kwargs['flying_zebra'].slow_axis, 'NANOVER')

    dets = [xs] + xrd_dets

    # Modify md
    if 'md' in kwargs:
        md = kwargs.pop('md')
    else:
        md = None
    md = get_stock_md(md)
    md['scan']['type'] = 'FLY_ANGLE_RC'
    kwargs['md'] = md

    def plan():
        yield from scan_and_fly_base(dets,
                                    start, stop, num,
                                    y_current,
                                    y_current,
                                    1,
                                    dwell,
                                    **kwargs)
    
    def finalize_scan():
        if return_to_start:
            yield from mov(nano_stage.th, start_th)
    
    yield from finalize_wrapper(plan(), finalize_scan)


# A static xrd measurement without changing energy or moving stages
@append_srx_kwargs_md
def static_xrd(num,
               dwell,
               xrd_dets,
               md=None,
               N_dark=10,
               vlm_snapshot=False,
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
    md = get_stock_md(md)
    md['scan']['type'] = 'STATIC_XRD'
    md['scan']['scan_input'] = [num, dwell]
    md['scan']['dwell'] = dwell                               
    md['scan']['start_time'] = ttime.ctime(ttime.time())
    # md['scan']['detectors'] = [d.name for d in dets]
    md_dets = dets
    if vlm_snapshot is True:
        md_dets = md_dets + [nano_vlm]
    get_det_md(md, md_dets)

    def at_scan(name, doc):
        time_rem = 30 # Over-estimate
        scanrecord.time_remaining.put(time_rem / 3600)
        scanreocrd.time_rem_str.put(time_rem_convert(time_rem))

    def finalize_scan():
        yield from abs_set(scanrecord.scanning, False)
        yield from abs_set(scanrecord.time_remaining, 0)
        yield from abs_set(scanrecord.time_rem_str, time_rem_convert(0))
        # scanrecord.scanning.put(False)
        # scanrecord.time_rem_str.put(time_rem_convert(0))

        # Reset xs and sclr1
        for obj, key, value in original_sigs:
            yield from abs_set(getattr(obj, key), value)

    # Live Callbacks
    livecallbacks = [LiveTable(['dexela_stats2_total'])]
    
    if plotme:
        livecallbacks.append(LivePlot('dexela_stats2_total'))

    @run_decorator(md=md)
    @vlm_decorator(vlm_snapshot, after=True)
    @dark_decorator(dets, N_dark=N_dark, shutter=shutter)  
    def plan():
        # Always check shutters to print banner
        yield from check_shutters(shutter, 'Open')
        yield from mod_count(dets, num, run_agnostic=True)
        if shutter: # Conditional check to avoid banner
            yield from check_shutters(shutter, 'Close')
    
    # Plan must be called to return the generators
    plan = finalize_wrapper(plan(), finalize_scan)
    # This actually runs the plan
    yield from subs_wrapper(plan, {'all' : livecallbacks,
                                   'start' : at_scan})


# Coordinating function for other reciprocal space mapping functions
@append_srx_kwargs_md
def point_rsm(start, stop, num,
              dwell,
              xrd_dets,
              rocking_axis='energy',
              relative=False,
              flying=False,
              md=None,
              **kwargs):

    # Parse inputs
    rocking_axis = rocking_axis.lower()
    if rocking_axis not in ['energy', 'wavelength', 'angle', 'theta', 'compucentric']:
        err_str = f"Rocking axis can only be in ['energy', 'wavelength', 'angle', 'theta', 'compucentric'] not {rocking_axis}"
        raise ValueError(err_str)

    if (rocking_axis in ['energy', 'wavelength']
        and flying is True):
        err_str = 'Flying with the monochromater is not yet supported for reciprocal space mapping.'
        raise NotImplementedError(err_str)
    if (rocking_axis in ['compucentric']
        and flying is True):
        err_str = 'Flying with compucentric pseudomotor is not yet supported for recirpcoal space mapping.'
        raise NotImplementedError(err_str)

    # Get relevant motor and adjust input units
    if rocking_axis == 'energy':
        rocking_motor = energy
        curr_pos = energy.energy.setpoint.get()
        
        # Convert to eV
        if curr_pos < 1e3:
            curr_pos *= 1e3
        if relative is True:
            # Less than 10 eV relative shifts are unlikely
            if start < 10 or stop > 10:
                warn_str = ("CAUTION: Relative inputs assumed to be given in keV and adjusted to eV."
                            + "\nIf this is a mistake, give absolute units in eV to avoid conversion.")
                print(warn_str)
                start *= 1e3
                stop *= 1e3
        else:
            # Absolute conversions are easier
            if start < 1e3 or stop > 1e3:
                start *= 1e3
                stop *= 1e3

    
    elif rocking_axis == 'wavelength':
        rocking_motor = energy
        curr_pos = energy.energy.setpoint.get()

        # Convert to eV
        if curr_pos < 1e3:
            curr_pos *= 1e3

        if relative is True:
            if start < 1e-4 or stop < 1e-4:
                warn_str = ("CAUTION: Relative inputs assumed to be given in nm and adjusted to Angstroms."
                            + "\nIf this is a mistake, give absolute units in Angstroms to avoid conversion.")
                print(warn_str)
                start *= 10
                stop *= 10
        else:
            # Convert from nm to A based on wavelength range
            if start < 0.45 or stop < 0.45:
                start *= 10
                stop *= 10
        
        # Convert to eV assuming A wavelength units
        hc = 12398.4        
        start = hc / start
        stop = hc / stop

    elif rocking_axis in ['angle', 'theta', 'compucentric']:
        if rocking_axis == 'compucentric':
            rocking_motor = comp_th
        else:
            rocking_motor = nano_stage.th
        curr_pos = nano_stage.th.user_setpoint.get()
        
        # Convert from deg to mdeg
        if curr_pos < 360:
            curr_pos *= 1e3

        if relative is True:
            if start < 60 or stop < 60:
                warn_str = ("CAUTION: Relative inputs assumed to be given in deg and adjusted to mdeg."
                            + "\nIf this is a mistake, give absolute units in mdeg to avoid conversion.")
                print(warn_str)
                start *= 1e3
                stop *= 1e3
        else:
            if start < 360 or stop < 360:
                start *= 1e3
                stop *= 1e3
    
    else: # Mostly for debugging
        err_str = f'Error parsing rocking_axis of {rocking_axis}'
        raise RuntimeError(err_str)

    # Make units relative
    if relative is True:
        start += curr_pos
        stop += curr_pos
    
    # Do the plan
    if rocking_axis in ['energy', 'wavelength']:
        yield from extended_energy_rsm(
                    start, stop, num,
                    dwell,
                    xrd_dets,
                    md=md
                    **kwargs)
    elif rocking_axis in ['compucentric']:
        yield from step_rsm_base(
                    start, stop, num,
                    dwell,
                    xrd_dets,
                    md=md,
                    **kwargs)
    elif rocking_axis in ['theta', 'angle']:
        if not flying:
            yield from step_rsm_base(
                        start, stop, num,
                        dwell,
                        xrd_dets,
                        md=md,
                        **kwargs)
        else:
            yield from flying_angle_rsm(
                        start, stop, num,
                        dwell,
                        xrd_dets,
                        md=md,
                        **kwargs)



### Older scan versions ####

    

@append_srx_kwargs_md
def energy_rocking_curve(e_low,
                         e_high,
                         e_num,
                         dwell,
                         xrd_dets,
                         md=None,
                         N_dark=10,
                         vlm_snapshot=True,
                         shutter=True,
                         peakup_flag=True,
                         plotme=False,
                         return_to_start=True):

    start_energy = energy.energy.readback.get()

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
    md = get_stock_md(md)
    md['scan']['type'] = 'ENERGY_RC'
    md['scan']['scan_input'] = [e_low, e_high, e_num, dwell]
    md['scan']['dwell'] = dwell
    # md['scan']['detectors'] = [d.name for d in dets]
    md_dets = dets
    if vlm_snapshot is True:
        md_dets = md_dets + [nano_vlm]
    get_det_md(md, md_dets)

    def at_scan(name, doc):
        time_rem = len(e_range) * (dwell + 4.25) # Some overhead for estimate
        scanrecord.time_remaining.put(time_rem / 3600)
        scanreocrd.time_rem_str.put(time_rem_convert(time_rem))

    def finalize_scan():
        yield from abs_set(scanrecord.scanning, False)
        yield from abs_set(scanrecord.time_remaining, 0)
        yield from abs_set(scanrecord.time_rem_str, time_rem_convert(0))
        # scanrecord.scanning.put(False)
        # scanrecord.time_rem_str.put(time_rem_convert(0))

    def time_per_point(name, doc, st=ttime.time()):
        if (doc[0] == "event_page"):
            if ('seq_num' in doc.keys()):
                scan_record.time_rem_str.put(convert_time_str(
                    ((doc['time'] - st) / doc['seq_num']) * # average time per point
                    (len(e_range) - doc['seq_num']) # remaining number of points
                ))

    # Live Callbacks
    livecallbacks = [LiveTable(['energy_energy', 'dexela_stats2_total'])]
    # livecallbacks.append(time_per_point)
    
    if plotme:
        livecallbacks.append(LivePlot('dexela_stats2_total', x='energy_energy'))

    # Move to center energy and perform peakup
    if peakup_flag:  # Find optimal c2_fine position
        print('Performing center energy peakup.')
        yield from mov(energy, e_cen)
        yield from peakup(shutter=shutter)
    
    @run_decorator(md=md)
    @vlm_decorator(vlm_snapshot, after=True)
    @dark_decorator(dets, N_dark=N_dark, shutter=shutter) 
    def plan():
        # Always check shutters to print banner
        yield from check_shutters(shutter, 'Open')
        yield from mod_list_scan(dets, energy, e_range, run_agnostic=True)
        if shutter: # Conditional check ot avoid banner
            yield from check_shutters(shutter, 'Close')

    # Plan must be called to return the generators
    plan = finalize_wrapper(plan(), finalize_scan)
    yield from subs_wrapper(plan, {'all' : livecallbacks,
                                   'start' : at_scan})

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
    
    en_current = energy.energy.readback.get()

    # Convert to keV. Not as straightforward as endpoint inputs
    if en_range > 5:
        warn_str = (f'WARNING: Assuming energy range of {en_range} '
                    + 'was given in eV.')
        print(warn_str)
        en_range /= 1000
    
    # Ensure energy.energy.readback.get() is reading correctly
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
                                  N_dark=10,
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

    for i, e_rc in enumerate(e_rcs):
        if i != 0:
            N_dark = 0
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


# WIP
# Re-write to encompass a single scan ID with intelligent vlm and dark-field support
# TODO: livecallbacks may fail switching back and forth...
@append_srx_kwargs_md
def continuous_energy_rocking_curve(e_low,
                                    e_high,
                                    e_num,
                                    dwell,
                                    xrd_dets,
                                    md=None,
                                    N_dark=0,
                                    vlm_snapshot=True,
                                    shutter=True,
                                    peakup_flag=True,
                                    plotme=False,
                                    return_to_start=True):
    raise NotImplementedError()
    
    start_energy = energy.energy.readback.get()

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
    md = get_stock_md(md)
    md['scan']['type'] = 'ENERGY_RC'
    md['scan']['scan_input'] = [e_low, e_high, e_num, dwell]
    md['scan']['dwell'] = dwell
    # md['scan']['detectors'] = [d.name for d in dets]
    md_dets = dets
    if vlm_snapshot is True:
        md_dets = md_dets + [nano_vlm]
    get_det_md(md, md_dets)

    # Live Callbacks
    livecallbacks = [LiveTable(['energy_energy', 'dexela_stats2_total'])]
    
    if plotme:
        livecallbacks.append(LivePlot('dexela_stats2_total', x='energy_energy'))

    @run_decorator(md=md)
    @vlm_decorator(vlm_snapshot, after=True)
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


@append_srx_kwargs_md
def angle_rocking_curve(th_low,
                        th_high,
                        th_num,
                        dwell,
                        xrd_dets,
                        md=None,
                        N_dark=10,
                        vlm_snapshot=True,
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
    md = get_stock_md(md)
    md['scan']['type'] = 'ANGLE_RC'
    md['scan']['scan_input'] = [th_low, th_high, th_num, dwell]
    md['scan']['dwell'] = dwell
    # md['scan']['detectors'] = [d.name for d in dets]
    md_dets = dets
    if vlm_snapshot is True:
        md_dets = md_dets + [nano_vlm]
    get_det_md(md, md_dets)

    def at_scan(name, doc):
        time_rem = len(e_range) * (dwell + 4.25) # Some overhead for estimate
        scanrecord.time_remaining.put(time_rem / 3600)
        scanreocrd.time_rem_str.put(time_rem_convert(time_rem))

    def finalize_scan():
        yield from abs_set(scanrecord.scanning, False)
        yield from abs_set(scanrecord.time_remaining, 0)
        yield from abs_set(scanrecord.time_rem_str, time_rem_convert(0))
        # scanrecord.scanning.put(False)
        # scanrecord.time_rem_str.put(time_rem_convert(0))

    def time_per_point(name, doc, st=ttime.time()):
        if (doc[0] == "event_page"):
            if ('seq_num' in doc.keys()):
                scan_record.time_rem_str.put(convert_time_str(
                    ((doc['time'] - st) / doc['seq_num']) * # average time per point
                    (len(th_range) - doc['seq_num']) # remaining number of points
                ))

    # Live Callbacks
    livecallbacks = [LiveTable(['nano_stage_th_user_setpoint', 'dexela_stats2_total']),
                     time_per_point]
    
    if plotme:
        livecallbacks.append(LivePlot('dexela_stats2_total', x='nano_stage_th_user_setpoint'))

    @run_decorator(md=md)
    @vlm_decorator(vlm_snapshot, after=True, position=(nano_stage.th, (th_high - th_low) / 2))
    @dark_decorator(dets, N_dark=N_dark, shutter=shutter) 
    def plan():
        # Always check shutters to print banner
        yield from check_shutters(shutter, 'Open')
        yield from mod_list_scan(dets, nano_stage.th, th_range, run_agnostic=True)
        if shutter: # Conditional check ot avoid banner
            yield from check_shutters(shutter, 'Close')

    # Plan must be called to return the generators
    plan = finalize_wrapper(plan(), finalize_scan)
    yield from subs_wrapper(plan, {'all' : livecallbacks,
                                   'start' : at_scan})

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

    dets = [xs] + xrd_dets

    # Modify md
    if 'md' in kwargs:
        md = kwargs.pop('md')
    else:
        md = None
    md = get_stock_md(md)
    md['scan']['type'] = 'FLY_ANGLE_RC'
    kwargs['md'] = md

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