print(f'Loading {__file__}...')

from bluesky.utils import FailedStatus

# SRX Enable PVs
sr_permit = EpicsSignalRO("SR-EPS{PLC:1}Sts:MstrSh-Sts")
srx_permit = EpicsSignalRO("XF:05ID-CT{}Prmt:Remote-Sel")
srx_enable = EpicsSignalRO("SR:C05-EPS{PLC:1}Sts:ID_BE_Enbl-Sts")

def auto_align(focus=0.5, all_checks=True):
    """
    Function for auto aligning the beamline.
    Must start with vlm marker on cross of knife edge features

    Parameters
    ----------
    focus : float or iterable, optional
        Target focus in microns. If given as an iterable, the target focus
        will be interpretted as (focus_y, focus_x) both in microns.
        By default both target values are 0.5 microns. 
    all_checks : bool, optional
        Flag for checking upstream optics.
        True by default and checks all upstream optics.
    """

    # Parse focus parameter
    if isinstance(focus, (float, int)):
        focus = (focus, focus)
    else:
        focus = (focus[0], focus[1])

    if all_checks:
        # Check we are on the commissioning proposal
        print(f"Checking proposal...", end="")
        if "beamline commissioning" in RE.md["proposal"]["title"].lower():
            print("PASS")
        else:
            print("FAIL!")
            raise RuntimeError("Please change the proposal!")

        # Check if we are enabled
        print("Checking beamline is enabled...", end="")
        if not (sr_permit.get(as_string=True) == "ENABLE" and
                srx_permit.get(as_string=True) == "Permitted" and
                srx_enable.get(as_string=True) == "Enabled"):
            print()
            raise RuntimeError("Beamline not fully enabled!")
        print("Enabled")

        # Close all shutters for safety
        print("Closing all shutters...")
        for shut in [shut_fe, shut_a, shut_b]:
            if shut.status.get() == 'Open':
                print(f'Closing {shut.name}...')
                try:
                    yield from mov(shut, 'Close')
                    yield from bps.sleep(5)
                except FailedStatus as e:
                    print(f'Cannot close {shut.name}!')
                    raise e

        # Close D-shutter
        try:
            yield from check_shutters('Close', True)
        except Exception as e:
            print('Unknown error encountered closing the D-shutter!')
            raise e
        print('All shutters closed!')

        # Change energy to 12 keV
        print('Moving energy to 12 keV...')
        yield from mov(energy, 12)
        # Requested positions
        sp_bragg, sp_c2_x, sp_u_gap = energy.energy_to_positions(12, 7, 0)
        # Actual positions
        rb_bragg = energy.bragg.user_readback.get()
        rb_c2_x = energy.c2_x.user_readback.get()
        rb_u_gap = energy.u_gap.gap.user_readback.get()
        if np.abs(sp_bragg - rb_bragg) > 0.001:
            raise RuntimeError(f'\nBragg angle of {rb_bragg:.6f} does not match setpoint of {rb_bragg:.6f}!')
        elif np.abs(sp_u_gap - rb_u_gap) > 1:
            raise RuntimeError(f'\nu_gap position of {rb_u_gap:.1f} does not match setpoint of {sp_u_gap:.1f}!')
        elif np.abs(sp_c2_x - rb_c2_x) > 0.1:
            raise RuntimeError(f'\nc2x position of {rb_c2_x:2f} does not match setpoint of {sp_c2_x:.2f}!')
        else:
            print('done!')

        # Check to make sure copper foil is in place
        print("Checking if Cu Foil is in place for BPM-B...", end="")
        if np.abs(bpm4_pos.y.user_readback.get() - 0) < 1:
            print('It is!')
        else:
            print('Wrong foil in place! Moving to Cu foil...')
            try:
                yield from mov(bpm4_pos.y, 0)
                print('Cu foil in place from BPM-B.')
            except Exception as e:
                print('Some unknown failure with changing BPM-B target foil to Cu...')
                raise e  

        # Open FE shutter
        print("Checking front-end shutter...")
        try:
            yield from mov(shut_fe, "Open")
            yield from bps.sleep(5)
            print("Front-end shutter opened")
        except FailedStatus as e:
            print("Cannot open front-end shutter! Is the A-hutch secured?")
            raise e
        
        # Read the background signal on bpm4
        blank_bpm4_current = bpm4.total_current.get()

        # Open A shutter
        print("Checking A shutter...")
        try:
            yield from mov(shut_a, "Open")
            yield from bps.sleep(5)
            print("A shutter opened")
        except FailedStatus as e:
            print("Cannot open A shutter! Is the B-hutch secured?")
            raise e

        # Read and check signal on bpm4 to ensure beam is passing through optics correctly
        print('Checking bpm4 to ensure beam is passing though...', end='')
        if np.abs(blank_bpm4_current - bpm4.total_current.get()) < 0.1:
            print('No current on bpm4. Is there an X-ray beam?')
            raise RuntimeError
        else:
            print('it is!')

        # Initial peakup for alignment
        print('Performing partial peakup...')
        try:
            yield from smart_peakup(detectors=[dcm.c2_pitch, bpm4], target_fields=['bpm4_total_current'], shutter=False)
            print('Success!')
        except Exception as e:
            print('Unknown error encountered with partial peakup function!')
            raise e
        
        # Set SSA to 50 um
        print('Setting SSA to 50 um gap...', end='')
        try:
            yield from mov(slt_ssa.h_gap, 0.05)
            yield from bps.sleep(5)
        except Exception as e:
            print('Unknown error encountered setting the SSA horizontal gap.')
            raise e
        # Check to confirm move really occured
        rb_slt_ssa_h_gap = slt_ssa.h_gap.readback.get()
        if np.abs(rb_slt_ssa_h_gap - 0.05) > 0.005:
            raise RuntimeError(f'SSA horizontal gap did not move correctly. Actual gap is {rb_slt_ssa_h_gap:.5f} um.')
        
        # Open B shutter
        print("Checking B shutter...")
        try:
            yield from mov(shut_b, "Open")
            yield from bps.sleep(5)
            print("B shutter opened")
        except FailedStatus as e:
            print("Cannot open B shutter! Is the D-hutch secured?")
            raise e

        # Remove all attenuators
        print('Removing all attenuators...')
        for attn_name in attenuators.component_names:
            try:
                yield from mov(getattr(attenuators, attn_name), 0)
            except Exception as e:
                print(f'\nUnknown error encountered removing {attn_name} attenuator!')
                raise e
        print('All attentuators removed!')
    
        # Peform full peakup
        print('Performing full peakup...')
        try:
            yield from peakup()
            print('Success!')
        except Exception as e:
            print('Unknown error encountered with full peakup function!')
            raise e

        # Optimize scalers
        print('Optimizing scaler pre-amplifier values...')
        try:
            yield from optimize_scalers()
        except Exception as e:
            print('Unknown error encountered optimizing scaler pre-amplifier values!')
            raise e

    # Check for sample!
    x0, y0 = nano_stage.topx.user_readback.get(), nano_stage.y.user_readback.get()
    over_x = nano_vlm.over.overlay_1.position_x.get()
    over_y = nano_vlm.over.overlay_1.position_y.get()
    over_dx, over_dy = 0, 0
    
    # Initial vertical focus checks
    print('Checking for vertical knife-edge position and focus...')
    yield from mov(nano_stage.topx, x0 - 50)
    try:
        # Find the feature
        print('Searching for vertical knife edge...')
        cent_y, _ = yield from knife_edge(nano_stage.sy, -45, 45, 1, 0.05)
        over_dy += cent_y
        print(f'Vertical knife edge found at {cent_y:.2f} um. Moving and centering stages at this new position!')
        yield from mov(nano_stage.sy, cent_y)
        yield from bps.sleep(1)
        yield from center_scanner()
        yield from bps.sleep(1)
        y0 = nano_stage.y.user_readback.get() # Not currently used

        # Measure the fwhm
        print('Measuring vertical focus...')
        cent_y, fwhm_y = yield from knife_edge(nano_stage.sy, -10, 10, 0.1, 0.05)
        over_dy += cent_y
        print(f'Vertical focus is {fwhm_y:.4f} um.')
        if fwhm_y < focus[0]:
            print(f'Vertical focus is less than desired {focus[0]} um!')
        else:
            print('Vertical focus is larger than desired focus. Manual focusing required.')
            pass

        # Final re-center scanner
        print(f'Final vertical knife edge found at {cent_y:.2f} um. Moving and centering stages at this new position!')
        yield from mov(nano_stage.sy, cent_y)
        yield from bps.sleep(1)
        yield from center_scanner()
        yield from bps.sleep(1)

    except Exception as e:
        print('Unknown exception encountered when checking the vertical focus!')
        raise e

    # Initial horizontal focus checks
    print('Checking for horizontal knife-edge position and focus...')
    yield from mov(nano_stage.topx, x0,
                   nano_stage.y, y0 + 50)
    try:
        # Find the feature
        print('Searching for horizontal knife edge...')
        cent_x, _ = yield from knife_edge(nano_stage.sx, -45, 45, 1, 0.05)
        over_dx += cent_x
        print(f'Horizontal knife edge found at {cent_x:.2f} um. Moving and centering stages at this new position!')
        yield from mov(nano_stage.sx, cent_x)
        yield from bps.sleep(1)
        yield from center_scanner()
        yield from bps.sleep(1)
        x0 = nano_stage.topx.user_readback.get() # Not currently used

        # Measure the fwhm
        print('Measuring horizontal focus...')
        cent_x, fwhm_x = yield from knife_edge(nano_stage.sx, -10, 10, 0.1, 0.05)
        over_dx += cent_x
        print(f'Horizontal focus is {fwhm_x:.4f} um.')
        if fwhm_x < focus[1]:
            print(f'Horizontal focus is less than desired {focus[1]} um!')
        else:
            print('Horizontal focus is larger than desired focus. Manual focusing required.')
            pass

        # Final re-center scanner
        print(f'Final horizontal knife edge found at {cent_x:.2f} um. Moving and centering stages at this new position!')
        yield from mov(nano_stage.sx, cent_x)
        yield from bps.sleep(1)
        yield from center_scanner()
        yield from bps.sleep(1)
        
    except Exception as e:
        print('Unknown exception encountered when checking the horizontal focus!')
        raise e

    # TODO: Check signs and cumulative moves
    vlm_scale = 0.345 # um/pixel
    yield from abs_set(nano_vlm.over.overlay_1.position_x,
                       over_x - int(np.round(vlm_scale / over_dx)), 
                       nano_vlm.over.overlay_1.position_y, 
                       over_y - int(np.round(vlm_scale / over_dy)))



    # Perform initial knife edges

    # # Slit scan on vertical KB
    # print('Focusing vertical KB...')
    # try:
    #     yield from focusKB('ver')
        
    #     # Move vertical KB mirror!

    # except Exception as e:
    #     print('Unknown error encountered when focusing vertical KB mirror!')
    #     raise e
    
    # # Slit scan on horizontal KB
    # print('Focusing horizontal KB...')
    # try:
    #     yield from focusKB('hor')

    #     # Move horizontal KB mirror!

    # except Exception as e:
    #     print('Unknown error encountered when focusing horizontal KB mirror!')
    #     raise e

    # # Perform final knife edges

    # # Close D-shutter to protect sample and prevent radiation damage
    # print('Closing D-shutter and wrapping up!')
    # try:
    #     yield from check_shutters('Close', True)
    # except Exception as e:
    #     print('Unknown error encountered closing the D-shutter!')
    #     raise e

        