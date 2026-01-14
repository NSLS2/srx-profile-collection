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

    # Record default motors in (x, y)
    fine_motors = (nano_stage.sx, nano_stage.sy)
    coarse_motors = (nano_stage.topx, nano_stage.y)

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

        # # Check if we are enabled
        # print("Checking beamline is enabled...", end="")
        # if not (sr_permit.get(as_string=True) == "ENABLE" and
        #         srx_permit.get(as_string=True) == "Permitted" and
        #         srx_enable.get(as_string=True) == "Enabled"):
        #     print()
        #     raise RuntimeError("Beamline not fully enabled!")
        # print("Enabled")
        
        # Check and wait for beamline to be enabled
        print("Checking if beamline is enabled...", end="")
        wait_iter = 0
        while True:
            if (sr_permit.get(as_string=True) == "ENABLE" and
                srx_permit.get(as_string=True) == "Permitted" and
                srx_enable.get(as_string=True) == "Enabled"):
                print('Enabled')
                break
            elif wait_iter > 60:
                print('Wait time for beamline to be enabled has surpased 1 hour!')
                raise RuntimeError('Beamline not fully enabled!')
            else:
                if wait_iter == 0:
                    print()
                ostr = f"Beamline not enabled. {wait_iter} minutes spent waiting."
                print(ostr, end='\r', flush=True)
                yield from bps.sleep(60)
                wait_iter += 1
                continue

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
        
    # Focusing
    for direction in ['vertical', 'horizontal']:
        # Get default values
        if 'ver' in direction:
            flag_dir = 'VER'
            overlay = nano_vlm.over.overlay_1.position_y
            focus_target = focus[0] # (VxH)              
            positioner = coarse_motors[0]
            scanner = fine_motors[1]
            jj_motor = jjslits.v_trans
            kb_trans_motor = nanoKB.v_y
            kb_pitch = nanoKB.v_pitch
            kb_fine = nanoKB.v_pitch_fine
            max_iter = 10

        elif 'hor' in direction:
            flag_dir = 'HOR'
            overlay = nano_vlm.over.overlay_1.position_x
            focus_target = focus[1] # (VxH)         
            positioner = coarse_motors[1]
            scanner = fine_motors[0]
            jj_motor = jjslits.j_trans
            kb_trans_motor = nanoKB.v_x
            kb_pitch = nanoKB.h_pitch
            kb_fine = nanoKB.h_pitch_fine
            max_iter = 5

        pos0 = positioner.user_setpoint.get()
        overlay_delta = 0

        # Convenience function to measure and center the beam
        def measure_and_center():
            # Fine scan to measure fwhm and better center the stages
            print(f'Measuring {direction} focus...')
            cent, fwhm, _ = yield from knife_edge(scanner, -10, 10, 0.1, 0.05)
            nonlocal overlay_delta
            overlay_delta += cent
            print(f'{direction.capitalize()} focus is {fwhm:.4f} um.')
            print(f'{direction.capitalize()} knife edge found at {cent:.2f} um. Moving and centering stages at this new position!')
            yield from mov(scanner, cent)
            yield from bps.sleep(1)
            yield from center_scanner()
            yield from bps.sleep(1)
            return fwhm
        
        # Function to protect alignment moves
        def limited_motor_move(motor,
                               rel_move,
                               min_move=0, max_move=np.inf,
                               staff_support=False):
            
            if np.abs(rel_move) < min_move:
                return False
            elif np.abs(rel_move) > max_move:
                err_str = (f'Motor {motor.name} requested move of {rel_move} {motor.motor_egu.get()}'
                           + f' is greater than the maximum allowed value of {max_move}.'
                           + '\nStaff support is required!')
                print(err_str)
                res = input("Should this motor be moved? ['y', 'n', or 'quit'] ")
            elif staff_support:
                err_str = (f'Motor {motor.name} requested move of {rel_move} {motor.motor_egu.get()}.'
                           + '\nThis move requires staff support!')
                print(err_str)
                res = input("Should this motor be moved? ['y', 'n', or 'quit'] ")
            else:
                print(f'Moving {motor.name} by {rel_move} {motor.motor_egu.get()}.')
                yield from mvr(motor, rel_move, timeout=10)
                return True

            # Check input values
            if res.lower() == 'y':
                print(f'Moving {motor.name} by {rel_move} {motor.motor_egu.get()}.')
                yield from mvr(motor, rel_move, timeout=10)
                return True
            elif res.lower() == 'n':
                return False
            else:
                raise RuntimeError(err_str)

        def end_iterations():
            # Re-center positioner
            yield from mov(positioner, pos0)
            # Move overlay 

            # TODO: Check signs and cumulative moves
            vlm_scale = 0.5 # um/pixel
            yield from abs_set(overlay,
                               overlay.get() - int(np.round(vlm_scale / overlay_delta)))    

        # Find the feature
        print(f'Searching for {direction} knife edge...')
        yield from mov(positioner, pos0 + 50)
        cent, _, _ = yield from knife_edge(scanner, -45, 45, 1, 0.05, plot=False)
        overlay_delta += cent
        print(f'{direction.capitalize()} knife edge found at {cent:.2f} um. Moving and centering stages at this new position!')
        yield from mov(scanner, cent)
        yield from bps.sleep(1)
        yield from center_scanner()
        yield from bps.sleep(1)
        
        # Fine scan to measure fwhm and center stages
        fwhm = yield from measure_and_center()
        if fwhm < focus_target:
            print(f'{direction.capitalize()} focus is less than desired {focus[0]} um!')
            yield from end_iterations()
            continue
        else:
            print(f'{direction.capitalize()} focus is larger than desired focus.')

        # Initial slit scan for alignment
        print(f'Initial {direction} slit scan for alignment and focusing...')
        beam_offset, kb_trans, defocus, pitch = yield from focusKB2(flag_dir)
        jj_move = yield from limited_motor_move(jj_motor,
                                                beam_offset,
                                                min_move=0.025, max_move=0.5,
                                                staff_support=True)
        if jj_move:
            kb_trans += beam_offset # Is this correct
        kb_move = yield from limited_motor_move(kb_trans_motor,
                                                kb_trans,
                                                min_move=0.025, max_move=0.5,
                                                staff_support=True)
        # Move sample position only for vertical
        if flag_dir == 'VER':
            z_move = yield from limited_motor_move(nano_stage.z,
                                                   defocus,
                                                   min_move=100, max_move=1000)
            if z_move:
                yield from limited_motor_move(nano_vlm_stage.z,
                                              defocus / 1000)
                
        # Re-run slit scan for any moves
        if any([jj_move, kb_move, z_move]):        
            # Fine scan to measure results
            fwhm = yield from measure_and_center()
            if fwhm < focus_target:
                print(f'{direction.capitalize()} focus is less than desired {focus[0]} um!')
                yield from end_iterations()
                continue
            else:
                print(f'{direction.capitalize()} focus is larger than desired focus.')
        
            beam_offset, kb_trans, defocus, pitch = yield from focusKB2(flag_dir)
        
        # Unpack pitch values
        pitch_angle, pitch_fine, pitch_defocus = pitch
        
        # Iterative focusing
        prev_fwhm = fwhm
        for iter in range(max_iter):
            # Linear correction
            if defocus > 100 and flag_dir == 'VER':
                z_move = yield from limited_motor_move(nano_stage.z,
                                                       defocus,
                                                       min_move=100, max_move=1000)
                if z_move:
                    yield from limited_motor_move(nano_vlm_stage.z,
                                                  defocus / 1000)
            # Quadratic correction
            else:
                if pitch_fine < 2:
                    pitch_move = yield from limited_motor_move(kb_fine,
                                                               pitch_fine,
                                                               min_move=0.05, max_move=2)
                else:
                    pitch_move = yield from limited_motor_move(kb_pitch,
                                                               pitch_angle,
                                                               min_move=0.005, max_move=0.05)
                # Move to new focal plane for vertical
                if pitch_move and flag_dir == 'VER':
                    z_move = yield from limited_motor_move(nano_stage.z,
                                                           pitch_defocus,
                                                           min_move=100, max_move=1000)
                    if z_move:
                        yield from limited_motor_move(nano_vlm_stage.z,
                                                      pitch_defocus / 1000)
            
            # Check if anything was adjusted and re-measure
            if not any([pitch_move, z_move]):
                note_str = (f'Adjustments for focusing in {direction} have become too small!'
                            + f'Final {direction} focus is {fwhm:.4f} nm.')
                print(note_str)
                yield from end_iterations()
                continue
            else:
                fwhm = yield from measure_and_center()
            
            # Perform all checks to break the iterations
            if fwhm < focus_target:
                print(f'{direction.capitalize()} focus is less than desired {focus[0]} um!')
                yield from end_iterations()
                continue
            elif fwhm > prev_fwhm * 1.5:
                err_str = (f'{direction.capitalize()} focus has increased from {prev_fwhm} to {fwhm} nm!'
                           'Some component focused in the wrong direction and staff support is requied!')
                raise RuntimeError(err_str)
            elif iter > max_iter:
                note_str = (f'Maximum number of iterations has been reached for {direction} focusing.'
                            + f'Final {direction} focus is {fwhm:.4f} um.')
                print(note_str)
                yield from end_iterations()
                continue
            else:
                print(f'{direction.capitalize()} focus is larger than desired focus. Continuing iterative focusing!')
                beam_offset, kb_trans, defocus, pitch = yield from focusKB2(flag_dir)
                pitch_angle, pitch_fine, pitch_defocus = pitch

    # Final measurements
    print(f'Fine vertical focus measurement...')
    pos0 = positioner.user_setpoint.get()
    yield from mov(coarse_motors[0], pos0 + 50)
    _, fwhm, _ = yield from knife_edge(fine_motors[1], -10, 10, 0.1, 0.05)
    print(f'Final vertical focus is {fwhm:.4f} um.')
    yield from mov(coarse_motors[0], pos0)

    print(f'Fine horizontal focus measurement...')
    pos0 = positioner.user_setpoint.get()
    yield from mov(coarse_motors[1], pos0 + 50)
    _, fwhm, _ = yield from knife_edge(fine_motors[0], -10, 10, 0.1, 0.05)
    print(f'Final horizontal focus is {fwhm:.4f} um.')
    yield from mov(coarse_motors[1], pos0)