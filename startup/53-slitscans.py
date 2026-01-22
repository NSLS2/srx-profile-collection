print(f'Loading {__file__}...')
import pandas as pd


def ssa_hcen_scan(start, stop, num,
                  shutter=True, fit_data=True):
    # Setup metadata
    scan_md = {}

    # Setup LiveCallbacks
    def my_factory(name):
        fig = plt.figure(num=name)
        ax = fig.gca()
        ax.cla()
        return fig, ax

    if "sumX" in xbpm2.read_attrs:
        key = "xbpm2_sumX"
    elif "total_current" in xbpm2.read_attrs:
        key = "xbpm2_total_current"
    else:
        raise KeyError

    liveplotx = 'slt_ssa_h_cen_readback'
    liveploty = key  # 'xbpm2_sumX'
    livetableitem = [liveplotx, liveploty]
    plotme = HackLivePlot(liveploty, x=liveplotx,
                          fig_factory=partial(my_factory, name='SSA Slit Scan'))
    livecallbacks = [LiveTable(livetableitem),
                     plotme]

    # Setup the scan
    @subs_decorator(livecallbacks)
    def myscan():
        ret = yield from scan([slt_ssa.h_cen, sclr1, xbpm2],
                              slt_ssa.h_cen,
                              start,
                              stop,
                              num,
                              md=scan_md)
        return ret

    # Record old position
    old_pos = slt_ssa.h_cen.position

    # Run the scan
    yield from check_shutters(shutter, 'Open')
    ret = yield from myscan()
    yield from check_shutters(shutter, 'Close')

    # Return to old position
    yield from mv(slt_ssa.h_cen, old_pos)

    # Do the fitting
    if fit_data:
        ssa_hcen_scan_fit_data(ret, plotme)


def ssa_hcen_scan_fit_data(scanuid,
                           plot=None,
                           plot_guess=False):
    # Get the scanid
    if type(scanuid) is str:
        uid = scanuid
    elif type(scanuid) is int:
        uid = c[scanuid].start["uid"]
    id_str = c[uid].start['scan_id']
    tbl = c[uid]["primary"]["data"]

    if "sumX" in xbpm2.read_attrs:
        key = "xbpm2_sumX"
    elif "total_current" in xbpm2.read_attrs:
        key = "xbpm2_total_current"
    else:
        raise KeyError

    # Get the data
    x_key = 'slt_ssa_h_cen_readback'
    y_key = key
    x = tbl[x_key].read()
    y = tbl[y_key].read()
    dydx = np.gradient(y, x)

    # Fit the data
    p_guess = [np.amax(y),
               0.017,
               0.5*(x[0] + x[-1]),
               np.mean(y[:3]),
               0]
    try:
        popt, _ = curve_fit(f_gauss, x, y, p0=p_guess)
    except:
        print('Raw fit failed.')
        popt = p_guess

    C = 2 * np.sqrt(2 * np.log(2))
    print(f'\nThe beam size is\t{C * popt[1]:.4f} mm')
    print(f'The center is at\t{popt[2]:.4f} mm.\n')

    # Plot variables
    x_plot = np.linspace(np.amin(x), np.amax(x), num=100)
    y_plot = f_gauss(x_plot, *popt)
    dydx_plot = np.gradient(y_plot, x_plot)

    # Check for plot
    if plot is None:
        _, ax = plt.subplots()
    else:
        ax = plot.ax

    # Display fit of raw data
    ax.cla()
    ax.plot(x, y, '*', label='Raw Data')
    if (plot_guess):
        ax.plot(x_plot, f_gauss(x_plot, *p_guess), '--', label='Guess fit')
    ax.plot(x_plot, y_plot, '-', label='Final fit')
    ax.set_title(f'Scan {id_str}')
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.legend()


def measure_SSA_FWHM():
    # TODO: Make the plots correctly
    # Currently, it will make the plot, but non-interactive
    # Also, try to fit intersection of 0.5 and normalized data, dd
    if "sumX" in xbpm2.read_attrs:
        key = "xbpm2_sumX"
    elif "total_current" in xbpm2.read_attrs:
        key = "xbpm2_total_current"
    else:
        raise KeyError

    uid = yield from subs_wrapper(scan([xbpm2], slt_ssa.h_gap, 0, 0.2, 41),
                                  LivePlot(key, x='slt_ssa_h_gap_readback'))

    tbl = c[uid]["primary"]["data"]
    x = tbl["slt_ssa_h_gap_readback"].read()
    d = tbl[key].read()
    dd = (d - np.amin(d)) / (np.amax(d) - np.amin(d))

    fig, ax = plt.subplots()
    ax.plot(x, dd)
    ax.plot(x, 0.5*np.ones(dd.shape), "--k")
    ax.set_xlabel("SSA Horizontal Gap [mm]")
    ax.set_ylabel("Normalized XBPM2 Signal")
    ax.set_title(f"Scan ID: {c[uid].start['scan_id']}")


def JJ_scan(motor, start, stop, num, shutter=True):
    # Setup metadata
    scan_md = {}
    get_stock_md(scan_md)

    # Setup LiveCallbacks
    liveplotfig1 = plt.figure()
    liveplotx = motor.name
    liveploty = im.name
    livetableitem = [motor.name, im.name, i0.name]
    livecallbacks = [LiveTable(livetableitem),
                     LivePlot(liveploty, x=liveplotx, fig=liveplotfig1)]

    # Setup the scan
    @subs_decorator(livecallbacks)
    def myscan():
        yield from scan([motor, sclr1],
                        motor,
                        start,
                        stop,
                        num,
                        md=scan_md)

    # Record old position
    old_pos = motor.position

    # Run the scan
    if (shutter):
        yield from mv(shut_b, 'Open')

    ret = yield from myscan()

    if (shutter):
        yield from mv(shut_b, 'Close')

    # Return to old position
    yield from mv(motor, old_pos)

    return ret

def slit_nanoflyscan(scan_motor, scan_start, scan_stop, scan_stepsize, acqtime,
                    slit_motor, slit_start, slit_stop, slit_stepsize, slitgap_motor, slit_gap,
                    normalize=False, scan_only=False, shutter=False,
                    plot=True, plot_guess=False):
    """
    scan_motor       motor   motor used for scan
    scan_start       float   starting position
    scan_stop        float   stopping position
    scan_stepsize    float   distance between data points
    slit_motor       motor   slit motor used for scan
    slit_start       float   starting position
    slit_stop        float   stopping position
    slit_stepsize    float   distance between data points
    slit_gap         float   the gap to close the slit to
    acqtime          float   counting time per step
    """

    if ('nano_stage_s' not in scan_motor.name):
        print('Please use a supported motor: nano_stage.sx or nano_stage.sy')
        print('Exiting...')
        return

    slit_orig_gap = slitgap_motor.user_readback.get()
    slit_orig_pos = slit_motor.user_readback.get()

    # always move the slit to a more negative position to avoid backlash
    
    yield from mov(slitgap_motor, 0)
    yield from mov(slit_motor, slit_start-0.2)
    # close the gap for scan

    yield from mov(slitgap_motor, slit_gap)

    # Need to convert stepsize to number of points
    snum = np.round((scan_stop - scan_start) / scan_stepsize) + 1
    lnum = np.round((slit_stop - slit_start) / slit_stepsize) + 1
    lnum_list = np.arange(lnum)

    slit_pos = slit_start + lnum_list * slit_stepsize

    # Define a figure factory
    def my_factory(name):
        fig = plt.figure(num=name)
        ax = fig.gca()
        return fig, ax

    # Open the shutter
    yield from check_shutters(True, 'Open') 
    
    # Making some assumptions here
    # Set the ROI pv
    # Assuming xs is the detector
    if hasattr(xs, 'channel01'):
        roi_pv = xs.channel01.mcaroi01.ts_total
        try:
            yield from abs_set(xs.cam.acquire, 'Done', timeout=1)
        except Exception as e:
            print('Timeout setting X3X to status \"Done\". Continuing...')
            print(e)
        try:
            # This erases the time-series array, otherwise we see the previous scan
            yield from abs_set(xs.channel01.mcaroi.ts_control, 2, wait=True, timeout=1)  # Stop time series collection
            yield from abs_set(xs.channel01.mcaroi.ts_control, 0, wait=True, timeout=1)  # Start/erase time series collection
            yield from abs_set(xs.channel01.mcaroi.ts_control, 2, wait=True, timeout=1)  # Stop time series collection
        except Exception as e:
            # Eating the exception
            print('The time-series did not clear correctly. Continuing...')
            print(e)

    livepopup = [
        SRX1DTSFlyerPlot(
            roi_pv.name,
            xstart=scan_start,
            xstep=scan_stepsize,
            xlabel=scan_motor.name,
        )
    ]

    if (scan_motor.name == 'nano_stage_sx'):
        y0 = nano_stage.sy.user_readback.get()
        @subs_decorator(livepopup)
        def _knife_plan():
           yield from  nano_scan_and_fly(scan_start, scan_stop, snum,
                                         y0, y0, 1, acqtime,
                                         shutter=False, plot=False, vlm_snapshot=False)
    else:
        x0 = nano_stage.sx.user_readback.get()
        @subs_decorator(livepopup)
        def _knife_plan():
            yield from nano_y_scan_and_fly(scan_start, scan_stop, snum,
                                           x0, x0, 1, acqtime,
                                           shutter=False, plot=False, vlm_snapshot=False)

    def _plan():
        uid = yield from _knife_plan()
        return uid
    
    uid_list = []
    for ii in slit_pos:
        yield from mov(slit_motor, ii)
        uid = yield from _plan()
        uid_list.append(uid)

    # Finish up
    yield from check_shutters(True, 'Close') 
    yield from mov(slit_motor, slit_orig_pos)
    yield from mov(slitgap_motor, slit_orig_gap)

    return uid_list


def slit_nanoflyscan_cal(scan_id_list=[], interp_range=None, orthogonality=False, plotme=None,
                         bin_low=934, bin_high=954, normalize=True):
   
    """
    This function takes a list of scan_id, process them one by one, then analyzes the optical abberations.

    scan_id          list    a list of slitscan ids
    interp_range     list    list of the fitting range, can drop some bad points
    orthogonality    bool    calculate the spherical abb or the astigmatism
    bin_low          integer the lower limit of ROI (usually Pt or Au), by default it's taking roi1
    bin_high         integer uppper limit of ROI
    normalize        bool    whether or not to normalize the data by the ion chamber, I0
    """

    numline = len(scan_id_list)
    slit_range = np.zeros((numline,), dtype=np.float64)
    line_pos_seq = np.zeros((numline,), dtype=np.float64)

    # Mirror parameters
    f_v = 295 * 1e+3  # um
    f_h = 125 * 1e+3
    theta_v = 3 # mrad
    theta_h = 3 # mrad
    L_h = 100 # mm  Length of horizontal mirror
#    conversion_factor_orth = np.array([-1.6581375e-4, 5.89e-4]) #unit: p/urad (V x H)
    conversion_factor_orth = np.array([-0.000165824, 5.89e-4]) #unit: p/urad (V x H)
    pitch_motion_conversion = np.array([225, 100]) # unit: mm (V x H)
    delta_fine_pitch = 0.0 # unit: um
 
    for idx, scan_id in enumerate(scan_id_list):
        bs_run = c[int(scan_id)]
        scan_id = int(bs_run.start['scan_id'])

        # Get the information from the previous scan
        ds = bs_run['stream0']['data']
        ds_keys = list(ds.keys())
        fluor_key = 'xs_fluor'
 
        if 'nano_stage_sx' in bs_run.start['scan']['fast_axis']['motor_name']:
            flag_dir = 'HOR'
            pos = 'enc1'
            slit_range[idx] = bs_run['baseline']['data']['jjslits_h_trans'][1]
        elif 'nano_stage_sy' in bs_run.start['scan']['fast_axis']['motor_name']:
            flag_dir = 'VER'
            pos = 'enc2'
            slit_range[idx] = bs_run['baseline']['data']['jjslits_v_trans'][1]
        else:
            print('Unknown motor')
            return

        if bin_low is None:
            bin_low = xs.channel01.mcaroi01.min_x.get()
        if bin_high is None:
            bin_high = xs.channel01.mcaroi01.max_x.get()
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
       
        numpts = x.shape
        
        # Error function with offset
        def f_offset_erf(x, A, sigma, x0, y0):
            x_star = (x - x0) / sigma
            return A * erf(x_star / np.sqrt(2)) + y0
        
        def f_two_erfs(x, A1, sigma1, x1, y1, A2, sigma2, x2, y2):
            x1_star = (x - x1) / sigma1
            x2_star = (x - x2) / sigma2
        
            f_combo = f_offset_erf(x, A1, sigma1, x1, y1) + f_offset_erf(x, A2, sigma2, x2, y2)
            return f_combo
        
        def line_fit(x, y):
            # p_guess = [0.5 * np.amax(y),
            #            1.000,
            #            0.5 * (x[0] + x[-1]) - 2.5,
            #            np.amin(y) + 0.5 * np.amax(y),
            #            -0.5 * np.amax(y),
            #            1.000,
            #            0.5 * (x[0] + x[-1]) + 2.5,
            #            np.amin(y) + 0.5 * np.amax(y)]       
            p_guess = [0.5 * np.amax(y),
                       .25,
                       0.5 * (x[0] + x[-1]) - 1.0,
                       0,
                       -0.5 * np.amax(y),
                       .25,
                       0.5 * (x[0] + x[-1]) + 1.0,
                       0]       
            try:
                print(p_guess)
                popt, _ = curve_fit(f_two_erfs, x, y, p0=p_guess)
            except:
                print('Raw fit failed.')
                popt = p_guess
        
            C = 2 * np.sqrt(2 * np.log(2))
            line_pos = (popt[2] + popt[6]) / 2
            print(f'The center beam position is {line_pos:.4f} um\n')
            return line_pos

        line_pos_seq[idx] = line_fit(x, y)

    if interp_range is None:
        interp_range = np.arange(numline)

    calpoly_fit = np.polyfit(slit_range[interp_range], line_pos_seq[interp_range]/1000, orthogonality+1, full=True)
    p = np.poly1d(calpoly_fit[0])
    line_plt = p(slit_range[interp_range])
    p2v_line_pos = np.max(line_pos_seq[interp_range])-np.min(line_pos_seq[interp_range])
 
    if flag_dir == 'VER':
        C_f = f_v
        C_theta = theta_v
        conversion_factor_orth = conversion_factor_orth[0]
        pitch_motion_conversion = pitch_motion_conversion[0]
    elif flag_dir == 'HOR':
        C_f = f_h
        C_theta = theta_h
        conversion_factor_orth = conversion_factor_orth[1]
        pitch_motion_conversion = pitch_motion_conversion[1] 
    else:
        print(f"Unknown direction: {flag_dir}")
        raise Exception


    defocus = -calpoly_fit[0][0] * C_f
    delta_theta = calpoly_fit[0][0] * C_theta
    actuator_move_h = delta_theta * L_h
    line_move_h = -2 * delta_theta * C_f * 1e-3
    print(f'Fitting results for {flag_dir} direction:')
    print(f'\tp is {calpoly_fit[0]}')
    # print(f'residual is {calpoly_fit[1]*1e+6} nm')
    print(f'\tP2V of line position is {p2v_line_pos:.4f} um')
    if (flag_dir == 'VER'):
        print(f'\tDefocus is {defocus:7.3f} um. Vkb correct by this amount.')
    elif (flag_dir == 'HOR'):
        print(f'\tDefocus is {defocus:7.3f} um. Hkb correct by this amount.')
        print(f'\tEquivalent to {delta_theta:7.6f} mrad. Hkb correct by this amount.')
        print(f'\tActuator should move by {actuator_move_h:7.3f} um.')
        print(f'\tLine feature should move {line_move_h:7.3f} um for h mirror pitch correction')

    #if (np.abs(defocus) < 100 or orthogonality == 1):
    if orthogonality == 1:
        delta_fine_pitch = -1*calpoly_fit[0][0]/conversion_factor_orth*1e-3*pitch_motion_conversion
        delta_theta_quad = calpoly_fit[0][0]/conversion_factor_orth
        delta_focal_plane_z = delta_theta_quad*1e-3/C_theta*C_f
        print('\nOrthagonality correction:')
        print(f'\tQuadratic term corresponds to pitch angle {delta_theta_quad:7.3f} urad.')
        print(f'\tQuadratic term corresponds to fine pitch move {delta_fine_pitch:7.3f} um.')
        print(f'\tQuadratic term corresponds to coarse Z {delta_focal_plane_z:7.3f} um.')


    if (plotme is None):
        fig, ax = plt.subplots()
    else:
        ax = plotme.ax
    ax.plot(slit_range, line_pos_seq/1000, 'ro', slit_range[interp_range], line_plt)
    ax.set_title(f'Scan {scan_id}')
    ax.set_xlabel(f'Slit Pos (mm)')
    ax.set_ylabel(f'Line Pos (mm)')

    fname = f'slitscan_{scan_id_list[0]}.png'
    root = '/home/xf05id1/current_user_data/knife_edge_scans/'
    try:
        os.makedirs(root, exist_ok=True)
        plt.savefig(root + fname, dpi=300)
    except:
        print('Could not save plot.')


def focusKB(direction, **kwargs):
    if 'hor' in direction.lower():
        kwargs.setdefault('scan_motor', nano_stage.sx)
        kwargs.setdefault('slit_motor', jjslits.h_trans)
        slit_range = 0.500
        kwargs.setdefault('slit_stepsize', 0.05)
        kwargs.setdefault('slitgap_motor', jjslits.h_gap)
        kwargs.setdefault('slit_gap', 0.05)
        N = 11
    elif 'ver' in direction.lower():
        kwargs.setdefault('scan_motor', nano_stage.sy)
        kwargs.setdefault('slit_motor', jjslits.v_trans)
        slit_range = 1.00
        kwargs.setdefault('slit_stepsize', 0.1)
        kwargs.setdefault('slitgap_motor', jjslits.v_gap)
        kwargs.setdefault('slit_gap',  0.10)
        N = 11
    else:
        print("This is for vertical or horizontal scans. Please choose one of these directions\n")

    kwargs.setdefault('scan_start', -8)
    kwargs.setdefault('scan_stop', 8)
    kwargs.setdefault('scan_stepsize', 0.20)
    kwargs.setdefault('acqtime', 0.200)
    
    slit_center = kwargs['slit_motor'].user_readback.get()
    kwargs.setdefault('slit_start', slit_center - 0.5 * slit_range)
    kwargs.setdefault('slit_stop', slit_center + 0.5 * slit_range)
    print(f'start from slit center: {slit_center}\n')

    # Definite a LivePlot for plotting later
    plotme = LivePlot('')

    # print(*kwargs)
    uids = yield from subs_wrapper(slit_nanoflyscan(**kwargs), plotme)

    # Fit the data
    # N = len(uids)
    scanids = np.linspace(-N, -1, num=N)
    slit_nanoflyscan_cal(scan_id_list=scanids, interp_range=scanids[2:-2].astype('int'), orthogonality=False, plotme=plotme)


def plot_slit_scans(start_id, N):
    fig, ax = plt.subplots()
    for scanid in [start_id + i for i in range(N)]:
        print(f'Loading {scanid}...', end='')
        h = db[int(scanid)]
        d = np.array(list(h.data('xs_fluor', stream_name='stream0', fill=True)))
        d_i0 = np.array(list(h.data('i0', stream_name='stream0', fill=True)))
        d = np.squeeze(d)
        d = np.sum(d[:, :, 934:954], axis=(-1, -2))
        d = np.squeeze(d)
        dd = d #/ np.squeeze(d_i0.T)

        ax.plot(dd, label=f"{scanid}")
        print('done')

    ax.legend()


def orth_slit_nanoflyscan_cal(scan_id_list=[], slit_range=[], from_RE=[], orthogonality=False, interp_range=None,                               bin_low=934, bin_high=954, normalize=True):
   
    """
    this function takes a list of scan_id, process them one by one, then analyzes the optical abberations.
    scan_id          list    a list of slitscan ids
    slit_range       list    a list of slit motor positions
    from_RE          list    plot handle
    orthogonality    bool    calculate the spherical abb or the astigmatism
    interp_range     list    list of the fitting range, can drop some bad points
    bin_low          integer the lower limit of ROI (usually Pt or Au), by default it's taking roi1
    bin_high         integer uppper limit of ROI
    """

    scan_id_list = np.array(scan_id_list)
    slit_range = np.array(slit_range)
    numline = scan_id_list.shape[0]
    line_pos_seq = np.zeros (int(numline))
    i = 0
    # Mirror parameters
    f_v = 295*1e+3  # um
    f_h = 125*1e+3
    theta_v = 3 #mrad
    theta_h = 3 #mrad
    conversion_factor_orth = np.array([-1.6581375e-4, 5.89e-4]) #unit: p/urad (V x H)
    pitch_motion_conversion = np.array([225, 100]) # unit: mm (V x H)
    delta_fine_pitch = 0.0 # unit: um

    for idx, scan_id in enumerate(scan_id_list):
        bs_run = c[int(scan_id)]
        scan_id = int(bs_run.start['scan_id'])

        # Get the information from the previous scan
        ds = bs_run['stream0']['data']
        ds_keys = list(ds.keys())
        fluor_key = 'xs_fluor'
 
        if 'nano_stage_sx' in bs_run.start['scan']['fast_axis']['motor_name']:
            flag_dir = 'HOR'
            pos = 'enc1'
        elif 'nano_stage_sy' in bs_run.start['scan']['fast_axis']['motor_name']:
            flag_dir = 'VER'
            pos = 'enc2'
        else:
            print('Unknown motor')
            return

        if bin_low is None:
            bin_low = xs.channel01.mcaroi01.min_x.get()
        if bin_high is None:
            bin_high = xs.channel01.mcaroi01.max_x.get()
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
       
        numpts = x.shape
       
        # Error function with offset
        def f_offset_erf(x, A, sigma, x0, y0):
            x_star = (x - x0) / sigma
            return A * erf(x_star / np.sqrt(2)) + y0
        
        def f_two_erfs(x, A1, sigma1, x1, y1, A2, sigma2, x2, y2):
            x1_star = (x - x1) / sigma1
            x2_star = (x - x2) / sigma2
        
            f_combo = f_offset_erf(x, A1, sigma1, x1, y1) + f_offset_erf(x, A2, sigma2, x2, y2)
            return f_combo
        
        def line_fit(x,y):
            p_guess = [0.5*np.amax(y),1.000,0.5*(x[0] + x[-1]) - 2.5,np.amin(y) + 0.5*np.amax(y),-0.5*np.amax(y),1.000,0.5*(x[0] + x[-1]) + 2.5,np.amin(y) + 0.5*np.amax(y)]       
            try:
                    # popt, _ = curve_fit(f_offset_erf, x, y, p0=p_guess)
                popt, _ = curve_fit(f_two_erfs, x, y, p0=p_guess)
            except:
                print('Raw fit failed.')
                popt = p_guess
        
            C = 2 * np.sqrt(2 * np.log(2))
            line_pos = (popt[2]+popt[6])/2
            print('\nThe center beam position is %f um' % ((popt[2]+popt[6])/2))
            return line_pos
        
        #line_pos_seq[i] = line_fit(pos_seq_plt, norm_line_seq[i,:])   
        line_pos_seq[i] = line_fit(x, y)  
        i=i+1 

    if interp_range is None:
        #if flag_dir == 'VER':
        #    interp_range = np.arange(numline)[:]
        #else:
        interp_range = np.arange(numline)

    calpoly_fit = np.polyfit(slit_range[interp_range], line_pos_seq[interp_range]/1000, orthogonality+1, full=True)
    p = np.poly1d(calpoly_fit[0])
    line_plt = p(slit_range[interp_range])
    p2v_line_pos = np.max(line_pos_seq[interp_range])-np.min(line_pos_seq[interp_range])
   
    if flag_dir == 'VER':
        C_f = f_v
        C_theta = theta_v
        conversion_factor_orth = conversion_factor_orth[0]
        pitch_motion_conversion = pitch_motion_conversion[0] 
    else:
        C_f = f_h
        C_theta = theta_h
        conversion_factor_orth = conversion_factor_orth[1]
        pitch_motion_conversion = pitch_motion_conversion[1] 


    print(f'p is {calpoly_fit[0]}')
    #print(f'residual is {calpoly_fit[1]*1e+6} nm')
    print(f'P2V of line position is {p2v_line_pos} um')
    defocus = -calpoly_fit[0][0] * C_f
    delta_theta = calpoly_fit[0][0] * C_theta
    line_move_h = -2 * delta_theta * C_f * 1e-3
    print('defocus is ' '{:7.3f}'.format(defocus), 'um. Vkb correct by this amount.')
    print('equivalent to ' '{:7.6f}'.format(delta_theta), 'mrad. Hkb correct by this amount.')
    #print('actuator should move by' '{:7.3f}'.format(actuator_move), 'um.')
    print('Line feature should move' '{:7.3f}'.format(line_move_h), 'um for h mirror pitch correction')

    if orthogonality == 1:
        delta_fine_pitch = calpoly_fit[0][0]/conversion_factor_orth*1e-3*pitch_motion_conversion
        delta_theta_quad = calpoly_fit[0][0]/conversion_factor_orth
        delta_focal_plane_z = delta_theta_quad*1e-3/C_theta*C_f
        print('quadratic term corresponds to pitch angle' '{:7.3f}'.format(delta_theta_quad), 'urad.')
        print('quadratic term corresponds to fine pitch move' '{:7.3f}'.format(delta_fine_pitch), 'um.')
        print('quadratic term corresponds to coarse Z ' '{:7.3f}'.format(delta_focal_plane_z), 'um.')


    fig, ax = plt.subplots()
    ax.plot(slit_range, line_pos_seq/1000, 'ro', slit_range[interp_range], line_plt)
    ax.set_title(f'scan {scan_id}')
    ax.set_xlabel(f'Slit Pos (mm)')
    ax.set_ylabel(f'Line Pos (mm)')

    fname = f'slitscan_{scan_id_list[0]}.png'
    root = '/home/xf05id1/current_user_data/'
    try:
        os.makedirs(root, exist_ok=True)
        plt.savefig(root + fname, dpi=300)
    except:
        print('Could not save plot.')

