# Run XRF-tomography
#
print(f'Loading {__file__}...')

import gc
import numpy as np
import matplotlib.pyplot as plt
import time as ttime
from scipy.ndimage import center_of_mass


# Convenience function for AMK
def haz_angles(a, b, n):
    th = np.linspace(a, b, num=n)
    th2 = np.concatenate((th[::2], th[-2::-2]))
    return th2


# Calculate the center of mass
def calc_com(run_start_uid, roi=None):
    print('Centering sample using center of mass...')

    # Get the header
    h = c[run_start_uid]
    scan_doc = h.start["scan"]
    
    # Get scan parameters
    [x0, x1, nx, y0, y1, ny, dt] = scan_doc["scan_input"]

    # Get the data
    flag_get_data = True
    t0 = ttime.monotonic()
    TMAX = 120  # wait a maximum of 60 seconds
    while flag_get_data:
        try:
            # d = list(h.data('fluor', stream_name='stream0', fill=True))
            # d = np.array(d)
            d = h["stream0"]["data"]["xs_fluor"].read()
            # d_I0 = list(h.data('i0', stream_name='stream0', fill=True))
            # d_I0 = np.array(d_I0)
            d_I0 = h["stream0"]["data"]["i0"].read()
            flag_get_data = False
        except:
            # yield from bps.sleep(1)
            if (ttime.monotonic() - t0 > TMAX):
                print('Data collection timed out!')
                print('Skipping center-of-mass correction...')
                return x0, x1, y0, y1
    # HACK to make sure we clear the cache.  The cache size is 1024 so
    # this would eventually clear, however on this system the maximum
    # number of open files is 1024 so we fail from resource exaustion before
    # we evict anything.
    # db._catalog._entries.cache_clear()
    # gc.collect()

    # Setup ROI
    if (roi is None):
        # Grab the values
        roi = [xs.channel01.mcaroi01.min_x.get(), xs.channel01.mcaroi01.min_x.get() + xs.channel01.mcaroi01.size_x.get()]
        
        # By default, do both low/high values reset to zero?
        if (roi[1] == 0):
            roi[1] = 4096
    d = np.sum(d[:, :, :, roi[0]:roi[1]], axis=(2, 3))
    d = d / d_I0
    d = d.T

    # Calculate center of mass
    if (scan_doc['fast_axis']['motor_name'] == 'nano_stage_sx'):
        (com_x, com_y)  = center_of_mass(d)  # for flying x scans
    elif (scan_doc['fast_axis']['motor_name'] == 'nano_stage_sy'):
        (com_y, com_x)  = center_of_mass(d)  # for y scans
    else:
        print('Not sure how data is oriented. Skipping...')
        return x0, x1, y0, y1
    com_x = x0 + com_x * (x1 - x0) / nx
    com_y = y0 + com_y * (y1 - y0) / ny
    # print(f'Center of mass X: {com_x}')
    # print(f'Center of mass Y: {com_y}')

    # Calculate new center
    extentX = x1 - x0
    old_center = x0 + 0.5 * extentX
    dx = old_center - com_x
    extentY = y1 - y0
    old_center_y = y0 + 0.5 * extentY
    dy = old_center_y - com_y

    # Check new location
    THRESHOLD = 0.50 * extentX
    if np.isfinite(com_x) is False:
        print('Center of mass is not finite!')
        new_center = old_center
    elif np.abs(dx) > THRESHOLD:
        print('New scan center above threshold')
        new_center = old_center
    else:
        new_center = com_x 
    new_center = np.round(new_center, 3)
    x0 = new_center - 0.5 * extentX
    x1 = new_center + 0.5 * extentX
    print(f'Old center: {old_center:.3f}')
    print(f'New center: {new_center:.3f}')
    print(f'  Difference: {dx:.3f}')

    THRESHOLD = 0.50 * extentY
    if np.isfinite(com_y) is False:
        print('Center of mass is not finite!')
        new_center_y = old_center_y
    elif np.abs(dy) > THRESHOLD:
        print('New scan center above threshold')
        new_center_y = old_center_y
    else:
        new_center_y = com_y 
    new_center_y = np.round(new_center_y, 3)
    y0 = new_center_y - 0.5 * extentY
    y1 = new_center_y + 0.5 * extentY
    print(f'Old center: {old_center_y:.3f}')
    print(f'New center: {new_center_y:.3f}')
    print(f'  Difference: {dy:.3f}')

    return x0, x1, y0, y1


# Define a function to call from the RunEngine
def nano_tomo(x0, x1, nx, y0, y1, ny, ct, th=None,
              th_offset=0,
              th_ind_start=0,
              centering_method='none',
              roi=None,
              fly_in_Y=False,
              extra_dets=[],
              shutter=True):
    # x0 = x starting point
    # x1 = x finish point
    # nx = number of points in x
    # y0 = y starting point
    # y1 = y finish point
    # ny = number of points in y
    # th = angles to scan at in degrees
    # th_offset = offset value to relate to rotation stage
    # th_ind_start = index of the angle to start at (zero-based)
    # centering_method = method used to account for sample motion and center
    #                    the sample
    #                    'none' = no correction
    #                    'com'  = center of mass
    # roi = [bin_low, bin_high] or None
    #       if [bin_low, bin_high], this will look at this ROI
    #       if None, this will grab the ROI bins from channel1 ROI 1

    # Set the angles for collection
    if (th is None):
        th = np.linspace(0, 180, 181)
    th = th + th_offset

    # Define callback for center of mass correction
    def cb_calc_com(name, doc):
        nonlocal x0, x1, y0, y1
        run_start_uid = doc['run_start']
        x0, x1, y0, y1 = calc_com(run_start_uid, roi=roi)

    # Open the shutter
    yield from check_shutters(shutter, 'Open')

    # Run the scan
    for i in th[th_ind_start:]:
        banner(f'Scanning at: {i:.3f} deg')

        # Rotate the sample
        if (nano_stage.th.egu == 'mdeg'):
            yield from mv(nano_stage.th, i * 1000)
        else:
            yield from mv(nano_stage.th, i)
        yield from bps.sleep(1)  # Give 1 second sleep to allow sample to settle
        
        # Run the scan/projection
        if fly_in_Y is False:
            myscan = nano_scan_and_fly(x0, x1, nx, y0, y1, ny, ct, extra_dets=extra_dets, shutter=False)
        else:
            myscan = nano_y_scan_and_fly(x0, x1, nx, y0, y1, ny, ct, extra_dets=extra_dets, shutter=False)

        if (centering_method == 'com'):
            myscan = subs_wrapper(myscan, {'stop' : cb_calc_com})
        yield from myscan

    # Close the shutter
    yield from check_shutters(shutter, 'Close')

    # Return to zero angle
    yield from mov(nano_stage.th, 0)


# Define a function to call from the RunEngine
def nano_Etomo(x0, x1, nx, y0, y1, ny, ct, th=None, energy_list=None,
               shutter=True, close_figs=True):
    if (th is None):
        print('Angle positions are required!')
        raise Exception
    if (energy_list is None):
        print('Energy points are required!')
        raise Exception

    # 1st dim: energy
    # 2nd dim: angles
    # Maybe change the 1st & 2nd dim, scan energy first is better?
    for ei in energy_list:
        # Change energy
        yield from mov(energy, ei)

        # Run tomography
        yield from nano_tomo(x0, x1, nx, y0, y1, ny, ct, th=th)

        # Close figures
        if close_figs:
            plt.close('all')

# Define a function to call from the RunEngine
def scan_coarsez(x0, x1, nx, y0, y1, ny, ct, z_list, 
              centering_method='none',
              roi=None,
              fly_in_Y=False,
              extra_dets=[],
              shutter=True):
    # x0 = x starting point
    # x1 = x finish point
    # nx = number of points in x
    # y0 = y starting point
    # y1 = y finish point
    # ny = number of points in y
    # th = angles to scan at in degrees
    # th_offset = offset value to relate to rotation stage
    # th_ind_start = index of the angle to start at (zero-based)
    # centering_method = method used to account for sample motion and center
    #                    the sample
    #                    'none' = no correction
    #                    'com'  = center of mass
    # roi = [bin_low, bin_high] or None
    #       if [bin_low, bin_high], this will look at this ROI
    #       if None, this will grab the ROI bins from channel1 ROI 1

    # Define callback for center of mass correction
    def cb_calc_com(name, doc):
        nonlocal x0, x1, y0, y1
        run_start_uid = doc['run_start']
        x0, x1, y0, y1 = calc_com(run_start_uid, roi=roi)

    # Open the shutter
    yield from check_shutters(shutter, 'Open')

    # Run the scan
    for z in z_list:
        banner(f'Scanning at: {z:.3f} um')

        if z < -2500:
            raise ValueError('Coarse z limit reached as safety!')

        # Rotate the sample
        yield from mv(nano_stage.z, z)
        yield from bps.sleep(1)  # Give 1 second sleep to allow sample to settle
        
        # Run the scan/projection
        if fly_in_Y is False:
            myscan = nano_scan_and_fly(x0, x1, nx, y0, y1, ny, ct, extra_dets=extra_dets, shutter=False, vlm_snapshot=False)
        else:
            myscan = nano_y_scan_and_fly(x0, x1, nx, y0, y1, ny, ct, extra_dets=extra_dets, shutter=False, vlm_snapshot=False)

        if (centering_method == 'com'):
            myscan = subs_wrapper(myscan, {'stop' : cb_calc_com})
        yield from myscan

    # Close the shutter
    yield from check_shutters(shutter, 'Close')

    # Return to zero angle
    yield from mov(nano_stage.th, 0)

