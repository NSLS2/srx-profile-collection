print(f'Loading {__file__}...')
import h5py
import time as ttime


# Run a knife-edge scan
def nano_knife_edge(motor, start, stop, stepsize, acqtime,
                    roi="Pt", normalize=True, use_trans=False,
                    scan_only=False, shutter=True, plot=True, plot_guess=False):
    """
    motor       motor   motor used for scan
    start       float   starting position
    stop        float   stopping position
    stepsize    float   distance between data points
    acqtime     float   counting time per step
    fly         bool    if the motor can fly, then fly that motor
    high2low    bool    scan from high transmission to low transmission
                        ex. start will full beam and then block with object (knife/wire)
    """

    # Need to convert stepsize to number of points
    num = np.round((stop - start) / stepsize) + 1

    # Set the roi
    setroi(1, roi)

    # Run the scan
    if (motor.name == 'nano_stage_sx'):
        fly = True
        pos = 'enc1'
        fluor_key = 'fluor'
        y0 = nano_stage.sy.user_readback.get()
        plotme = LivePlot('')
        @subs_decorator(plotme)
        def _plan():
            yield from nano_scan_and_fly(start, stop, num,
                                         y0, y0, 1, acqtime,
                                         shutter=shutter)
        yield from _plan()
    elif (motor.name == 'nano_stage_sy'):
        fly = True
        pos = 'enc2'
        fluor_key = 'fluor'
        x0 = nano_stage.sx.user_readback.get()
        plotme = LivePlot('')
        @subs_decorator(plotme)
        def _plan():
            yield from nano_y_scan_and_fly(start, stop, num,
                                           x0, x0, 1, acqtime,
                                           shutter=shutter)
        yield from _plan()
    elif (motor.name == 'nano_stage_x'):
        fly = False
        pos = motor.name
        fluor_key = 'xs2_channel1'
        y0 = nano_stage.y.user_readback.get()
        dets = [xs2, sclr1]
        yield from abs_set(xs2.total_points, num)
        livecallbacks = [
            LiveTable(
                [
                    motor.name,
                    xs2.channel1.rois.roi01.value.name
                ]
            )    
        ]
        
        livecallbacks.append(
            LivePlot(
                xs2.channel1.rois.roi01.value.name,
                motor.name
            )
        )

        if (shutter):
            yield from mov(shut_b, 'Open')
        yield from subs_wrapper(scan(dets, motor, start, stop, num),
                                {'all' : livecallbacks})
        if (shutter):
            yield from mov(shut_b, 'Close')
    elif (motor.name == 'nano_stage_y'):
        fly = False
        pos = motor.name
        fluor_key = 'xs2_channel1'
        x0 = nano_stage.x.user_readback.get()
        dets = [xs2, sclr1]
        yield from abs_set(xs2.total_points, num)
        livecallbacks = [LiveTable([motor.name,
                                    xs2.channel1.rois.roi01.value.name])]
        livecallbacks.append(LivePlot(xs2.channel1.rois.roi01.value.name,
                                      motor.name))
        if (shutter):
            yield from mov(shut_b, 'Open')
        yield from subs_wrapper(scan(dets, motor, start, stop, num),
                                {'all' : livecallbacks})
        if (shutter):
            yield from mov(shut_b, 'Close')
    else:
        print(f'{motor.name} is not implemented in this scan.')
        return

    # Do not do fitting, only do the scan
    if (not scan_only):
        return plot_knife_edge(scanid=db[-1].start['scan_id'], plot_guess=False, plotme=plotme)


# Make nice alias
knife_edge = nano_knife_edge


# Written quickly
def plot_knife_edge(scanid=-1, fluor_key='xs_fluor', use_trans=False, normalize=True, plot_guess=False,
                    bin_low=None, bin_high=None, plotme=None):
    # Get the scanid
    bs_run = c[int(scanid)]
    id_str = bs_run.start['scan_id']
    if 'FLY' in bs_run.start['scan']['type']:
        fly = True
    else:
        fly = False

    fast_axis = bs_run.start['scan']['fast_axis']['motor_name']

    try:
        if (fast_axis=='nano_stage_sx'):
            pos = 'enc1'
        else:
            pos = 'enc2'
    except:
        print('Not a knife-edge scan')
        return

    # Get the information from the previous scan
    ds = bs_run['stream0']['data']
    ds_keys = list(ds.keys())
    
    # Get the data
    if (use_trans == True):
        y = ds['it'].read() / ds['im'].read()
    else:
        if bin_low is None:
            bin_low = xs.channel01.mcaroi01.min_x.get()
        if bin_high is None:
            bin_high = xs.channel01.mcaroi01.min_x.get() + xs.channel01.mcaroi01.size_x.get()
        d = ds[fluor_key][..., bin_low:bin_high].sum(axis=(-2, -1)).squeeze()
        if 'i0' in ds_keys:
            I0 = ds['i0'].read().squeeze()
        elif 'sclr_i0' in ds_keys:
            I0 = ds['sclr_i0'].read().squeeze()
        else:
            raise KeyError
        if (normalize):
            y = np.array(d / I0).astype(np.float64)
        else:
            y = d.astype(np.float64)
    x = ds[pos].read().squeeze().astype(np.float64)
    dydx = np.gradient(y, x)

    # EJM better guess
    p_guess = [0.5*np.amax(y),
               0.500 / 2.3548, # 500 nm fwhm to sigma
               x[np.argmax(dydx)],
               np.amin(y),
               -0.5*np.amax(y),
               0.500 / 2.3548, # 500 nm fwhm to sigma
               x[np.argmin(dydx)],
               np.amin(y)]

    try:
        # popt, _ = curve_fit(f_offset_erf, x, y, p0=p_guess)
        popt, _ = curve_fit(f_two_erfs, x, y, p0=p_guess)
    except:
        print('Raw fit failed.')
        popt = p_guess

    C = 2 * np.sqrt(2 * np.log(2))
    cent_position = (popt[2]+popt[6])/2
    feature_size = np.abs(popt[2] - popt[6])
    print(f'The beam size is {C * popt[1]:.4f} um')
    print(f'The beam size is {C * popt[5]:.4f} um')

    #print(f'\nThe left edge is at\t{popt[2]:.4f}.')
    #print(f'The right edge is at\t{popt[6]:.4f}.')
    print(f'The center is at\t{cent_position:.4f}.')
    print(f'Feature size is\t\t{feature_size:.4f} um')

    # Plot variables
    x_plot = np.linspace(np.amin(x), np.amax(x), num=100)
    y_plot = f_two_erfs(x_plot, *popt)
    # y_plot = f_offset_erf(x_plot, *popt)
    dydx_plot = np.gradient(y_plot, x_plot)

    # Display fit of raw data
    if (plotme is None):
        fig, ax = plt.subplots()
    else:
        ax = plotme.ax

    ax.cla()
    ax.plot(x, y, '*', label='Raw Data')
    if (plot_guess):
        ax.plot(x_plot, f_two_erfs(x_plot, *p_guess), '--', label='Guess fit')
    ax.plot(x_plot, y_plot, '-', label='Final fit')
    ax.set_title(f'Scan {id_str}')
    ax.set_xlabel(fast_axis)
    if (normalize):
        ax.set_ylabel('Normalized ROI Counts')
    else:
        ax.set_ylabel('ROI Counts')
    ax.legend()

    return cent_position, C * np.mean([popt[1], popt[5]])
