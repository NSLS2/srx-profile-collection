print(f'Loading {__file__}...')

import numpy as np
import string
import matplotlib.pyplot as plt
import subprocess
import scipy as sp
import scipy.optimize
import lmfit
import threading
from scipy.optimize import curve_fit

from bluesky.callbacks.mpl_plotting import QtAwareCallback

# For smart_peakup
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from bluesky.utils import Msg
from bluesky import utils

'''

This program provides functionality to calibrate HDCM energy:
    With provided XANES scan rstuls, it will calculate their edge DCM location
    It will then fit the E vs Bragg RBV with four values,
    provide fitting results: dtheta, dlatticeSpace

#1. collected xanes at 3-5 different energies
    e.g. Ti(5 keV), Fe(7 keV), Cu (9 keV), Se (12 keV)
    They can be in XRF mode or in transmission mode;
    note the scan id in bluesky
#2. setup the scadid in scanlogDic dictionary
    scanlogDic = {'Fe': 264, 'Ti': 265, 'Cr':267, 'Cu':271, 'Se': 273}
#3. pass scanlogDic to braggcalib()
    braggcalib(scanlogDic = scanlogDic, use_xrf = True)
    currently, the xanes needs to be collected on roi1 in xrf mode
'''

def mono_calib(Element, acqtime=1.0, peakup=True):
    """
    SRX mono_calib(Element, acqtime=1.0, peakup=True)

    Go to the edge of the specified element, do a peakup, setroi, and automatic perform a xanes scan (+-50eV) on the specified element

    Parameters
    ----------
    Element : str

    Returns
    -------
    None

    Examples
    --------
    >>> mono_calib('V')

    """

    getemissionE(Element)
    EnergyX = getbindingE(Element)
    # energy.move(EnergyX)
    yield from mov(energy, EnergyX)
    setroi(1, Element)
    if peakup:
        yield from bps.sleep(5)
        yield from peakup()
    yield from xanes_plan(erange=[EnergyX-100,EnergyX+50],
                          estep=[1.0],
                          samplename=f'{Element}Foil',
                          filename=f'{Element}Foilstd',
                          acqtime=acqtime,
                          shutter=True)
def scan_all_foils(el_list = ['V', 'Cr', 'Fe', 'Cu', 'Zn', 'Se']):
    pos = {'V' : (-770, 900, 0.02, 0),#ssa=0.02, no filter
           'Cr': (8230, 900, 0.01, 0),#ssa=0.01, no filter
           'Fe': (26230, 900, 0.05, 2),#ssa=0.05, filter2 in
           'Cu': (8230, 9900, 0.01, 5),#ssa=0.01, filter2 in
           'Zn': (17230, 9900, 0.1, 5),#ssa = 0.1, filter3 in
           'Se': (26230, 9900, 0.015, 5)}#ssa=0.015, filter2+3 in
    for el in el_list:
        yield from mv(slt_ssa.h_gap, pos[el][2]) 
        if pos[el][3] == 2:
            yield from mv(attenuators.Cu_shutter, 1)
        elif pos[el][3] == 3:
            yield from mv(attenuators.Si_shutter, 1)
        elif pos[el][3] == 5:
            yield from mv(attenuators.Cu_shutter, 1)
            yield from mv(attenuators.Si_shutter, 1)
         

        yield from bps.sleep(2)
        yield from mov(nano_stage.x, pos[el][0], nano_stage.y, pos[el][1])
        yield from mono_calib(el, peakup=True, peakup_calib=False)
        ## just open up all the shutters
        yield from mv(attenuators.Cu_shutter, 0)
        yield from mv(attenuators.Si_shutter, 0)
        yield from mv(slt_ssa.h_gap, 0.05) 
       


def scanderive(xaxis, yaxis, ax, xlabel='', ylabel='', title='', edge_ind=None):
    dyaxis = np.gradient(yaxis, xaxis)

    if edge_ind is None:
        # edge_ind = dyaxis.argmin()
        # Hack to remove endpoints by EJM on 20250915
        edge_ind = dyaxis[1:-1].argmin() + 1

    edge = xaxis[edge_ind]
    # if xaxis[-1] < 1000:
    #     edge = xaxis[dyaxis.argmin()]
    # else:
    #     edge = xaxis[dyaxis.argmax()]

    # fig, ax = plt.subplots()
    # p = plt.plot(xaxis, dyaxis, '-')
    ax.plot(xaxis, dyaxis, '-')
    # ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    p = ax.plot(edge, dyaxis[edge_ind], '*r', markersize=25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return p, xaxis, dyaxis, edge_ind


def find_edge(scanid, use_xrf=True, element='', norm_key="i0"):

    # Allow for step or fly scans
    h = c[scanid]
    if "XAS_STEP" in h.start["scan"]["type"]:
        tbl = h["primary"]["data"]
        braggpoints = tbl["energy_bragg"].read()
        energypoints = tbl["energy_energy_setpoint"].read()
        if "i0" in norm_key.lower():
            i0 = tbl["sclr_i0"].read()
        elif "im" in norm_key.lower():
            i0 = tbl["sclr_im"].read()
        else:
            raise KeyError(f"{norm_key} not recognized!")
        # XRF
        if use_xrf:
            xrf = np.sum([tbl[f"xs_channel0{ind}_fluor"] for ind in range(1, 8)], axis=0)
        else:
            it = tbl["sclr_it"].read().astype(float)
    elif "XAS_FLY" in h.start["scan"]["type"]:
        tbl = h["scan_001"]["data"]
        energypoints = tbl["energy"].read()[0]
        if 'energy_bragg' in tbl:
            braggpoints = tbl["energy_bragg"].read()[0]
        else:
            warn_str = ('WARNING: Bragg was not found for XAS_FLY scanning. Back-calculating Bragg '
                       + 'from recorded energy components using the current id_fly_device hdcm_parameters.')
            print(warn_str)
            d111 = id_fly_device.hdcm_parameters.d111.get()
            delta_bragg = id_fly_device.hdcm_parameters.delta_bragg.get()
            braggpoints = (np.arcsin((ANG_OVER_EV / (energypoints / 1e3)) / (2 * d111)) / np.pi * 180 - delta_bragg)

        if "i0" in norm_key.lower():
            i0 = tbl["i0"].read()[0]
        elif "im" in norm_key.lower():
            i0 = tbl["im"].read()[0]
        else:
            raise KeyError(f"{norm_key} not recognized!")
        # XRF
        if use_xrf:
            xrf = np.sum([tbl[f"xs_id_mono_fly_channel0{ind}"][0] for ind in range(1, 8)], axis=0)
        else:
            it = tbl["it"].read()[0].astype(float)
    else:
        raise TypeError(f'Scan of {h.start["scan"]["type"]} not supported for Bragg calibration.')

    # Parse down to ROI
    if use_xrf:
        if element == '':
            raise ValueError('Please send the element name')

        el_line = getemissionE(element, 'ka1')
        el_slice = slice(int(el_line * 1e2) - 10, int(el_line * 1e2) + 10)
        mu = np.sum(xrf[:, el_slice], axis=1, dtype=float)
        mu /= i0.astype(float)
    else:
        tau = it / i0.astype(float)
        norm_tau = (tau - tau[0]) / (tau[-1] - tau[0])
        mu = -1 * np.log(np.abs(norm_tau))
    
    # print(f'{braggpoints.shape=}')
    # print(f'{energypoints.shape=}')
    # print(f'{mu.shape=}')

    # if use_xrf is False:
    #     tau = it / i0
    #     norm_tau = (tau - tau[0]) / (tau[-1] - tau[0])
    #     mu = -1 * np.log(np.abs(norm_tau))
    # else:
    #     if (element == ''):
    #         raise ValueError('Please send the element name')
    #     else:
    #         mu = np.zeros((tbl["xs_channel01_mcaroi01_total_rbv"].shape))
    #         for i in range(1, 8):
    #             ch_name = f"xs_channel{i:02}_mcaroi01_total_rbv"
    #             mu += tbl[ch_name].read()
    #         mu /= i0

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(element)
    p, xaxis, yaxis, edge_ind = scanderive(braggpoints, mu, ax1, xlabel='Bragg [deg]', ylabel='Gradient')
    edge = braggpoints[edge_ind]
    Ep, Exaxis, Eyaxis, edge_ind = scanderive(energypoints, mu, ax2, xlabel='Energy [eV]', edge_ind=edge_ind)
    Eedge = energypoints[edge_ind]

    return p, xaxis, yaxis, edge, Eedge


def braggcalib(scanlogDic={}, use_xrf=True, man_correction={}):
    if scanlogDic == {}:
        raise ValueError("scanlogDic cannot be empty!")

    fitfunc = lambda pa, x: (12.3984 /
                             (2 * pa[0] * np.sin((x + pa[1]) * np.pi / 180)))
    errfunc = lambda pa, x, y: fitfunc(pa, x) - y

    energyDic = {'Cu': 8.979, 'Se': 12.658, 'Zr': 17.998, 'Nb': 18.986,
                 'Ti': 4.966, 'Cr': 5.989, 'Co': 7.709, 'V': 5.465,
                 'Ni': 8.333, 'Fe': 7.112, 'Mn': 6.539, 'Zn': 9.659,
                 'Mo': 20.000}

    BraggRBVDic = {}
    EnergyRBVDic = {}
    fitBragg = []
    fitEnergy = []

    for element in scanlogDic:
        print(f"{element}: {scanlogDic[element]}")
        current_scanid = scanlogDic[element]

        p, xaxis, yaxis, edge, Eedge = find_edge(scanid=current_scanid,
                                                 use_xrf=use_xrf,
                                                 element=element)

        BraggRBVDic[element] = round(edge, 6)
        EnergyRBVDic[element] = round(Eedge, 6)

        # plt.show(p)
        fitBragg.append(BraggRBVDic[element])
        fitEnergy.append(energyDic[element])

        if element in man_correction:
            fitBragg[-1] = man_correction[element]
            man_correction.pop(element)

        # print('Edge position is at Bragg RBV: \t ', BraggRBVDic[element])
        print('Edge position is at Bragg RBV: \t ', fitBragg[-1])
        # print('Edge position is at Energy RBV (not calibrated): \t ', EnergyRBVDic[element])
        # print('Difference in Energy in Edge position: \t', EnergyRBVDic[element]-energyDic[element]*1000)


    fitEnergy = np.sort(fitEnergy)
    fitBragg = np.sort(fitBragg)[-1::-1]

    guess = [3.1356, 0.32]
    fitted_dcm, success = sp.optimize.leastsq(errfunc,
                                              guess,
                                              args=(fitBragg, fitEnergy))

    # print('(111) d spacing:\t', fitted_dcm[0])
    # print('Bragg RBV offset:\t', fitted_dcm[1])
    # print('Success:\t', success)

    newEnergy = fitfunc(fitted_dcm, fitBragg)

    print(fitBragg)
    print(newEnergy)

    fig, ax = plt.subplots()
    ax.plot(fitBragg, fitEnergy, 'b^', label='Raw scan')
    bragg = np.linspace(fitBragg[0], fitBragg[-1], 200)
    ax.plot(bragg, fitfunc(fitted_dcm, bragg), 'k-', label='Fitting')
    ax.legend()
    ax.set_xlabel('Bragg RBV (deg)')
    ax.set_ylabel('Energy (keV)')

    # pyplot.show()
    print('(111) d spacing:', fitted_dcm[0])
    print('Bragg RBV offset:', fitted_dcm[1])


class PairedCallback(QtAwareCallback):
    def __init__(self, scaler, dcm_c2_pitch_name, pitch_guess, gauss_height, *args, **kwargs):
        super().__init__(use_teleporter=kwargs.pop('use_teleporter', None))
        self.__setup_lock = threading.Lock()
        self.__setup_event = threading.Event()

        def setup():
            fig, ax = plt.subplots()
            self.ax = ax
            # fig.canvas.set_window_title('Peakup')
            # To remove deprecation warning use below. Needs testing
            fig.canvas.manager.set_window_title('Peakup')
            self.ax.clear()
            # Setup LiveCallbacks
            self.live_plot = LivePlot(scaler, dcm_c2_pitch_name,
                                      linestyle='', marker='*', color='C0',
                                      label='raw',
                                      ax=self.ax,
                                      use_teleporter=False)

            # Setup LiveFit
            # f_gauss(x, A, sigma, x0, y0, m)
            model = lmfit.Model(f_gauss, ['x'])
            init_guess = {'A': lmfit.Parameter('A', gauss_height, min=0),
                          'sigma': lmfit.Parameter('sigma', 0.001, min=0),
                          'x0': lmfit.Parameter('x0', pitch_guess),
                          'y0': lmfit.Parameter('y0', 0, min=0),
                          'm': lmfit.Parameter('m', 0, vary=False)}
            self.lf = LiveFit(model, scaler, {'x': dcm_c2_pitch_name}, init_guess)
            self.lpf = LiveFitPlot(self.lf, ax=self.ax, color='r', use_teleporter=False, label='Gaussian fit')

        self.__setup = setup

    def start(self, doc):
        self.__setup()
        self.live_plot.start(doc)
        self.lpf.start(doc)
        super().start(doc)

    def descriptor(self, doc):
        self.live_plot.descriptor(doc)
        self.lpf.descriptor(doc)
        super().descriptor(doc)

    def event(self, doc):
        self.live_plot.event(doc)
        self.lpf.event(doc)
        super().event(doc)

    def event_page(self, doc):
        self.live_plot.event_page(doc)
        self.lpf.event_page(doc)
        super().event_page(doc)

    def stop(self, doc):
        self.live_plot.stop(doc)
        self.lpf.stop(doc)
        super().stop(doc)


@parameter_annotation_decorator({
    "parameters": {
        "motor": {"default": "dcm.c2_fine"},
        "detectors": {"default": ['bpm3', 'bpm4', 'xbpm2']},
    }
})
def smart_peakup(start=None,
                 min_step=0.005,
                 max_step=0.50,
                 *,
                 shutter=True,
                 motor=dcm.c2_fine,
                 detectors=[dcm.c2_pitch, bpm4, xbpm2],
                 # target_fields=['bpm4_total_current', 'xbpm2_sumT'],
                 target_fields=['bpm4_total_current', 'xbpm2_total_current'],
                 MAX_ITERS=100,
                 md=None,
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
    _md = {'plan_name': 'smart_peakup',
           'hints': {},
           }
    _md = get_stock_md(_md)
    _md['scan']['type'] = 'PEAKUP'
    # _md['scan']['detectors'] = [det.name for det in detectors]
    _md['scan']['motors'] = [motor.name]
    _md['scan']['plan_args'] = {
                         'start': start,
                         'min_step': min_step,
                         'max_step': max_step,
                         }
    _md.update(md or {})
    get_det_md(_md, list(detectors))

    try:
        dimensions = [(motor.hints['fields'], 'primary')]
    except (AttributeError, KeyError):
        pass
    else:
        _md['hints'].setdefault('dimensions', dimensions)

    # Visualization
    livecb = []
    if verbose is False:
        livecb.append(LiveTable([motor.readback.name] + target_fields))

    # Need to add LivePlot, or LiveTable
    @bpp.stage_decorator(list(detectors) + [motor])
    @bpp.run_decorator(md=_md)
    @bpp.subs_decorator(livecb)
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
                yield Msg('create', None, name='primary')
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

    return (yield from smart_max_core(start))


# TEMP
def doms_peakup():
    yield from check_shutters(True, 'Open')
    yield from smart_peakup(detectors=[dcm.c2_pitch, bpm4, sclr1], target_fields=['bpm4_total_current', 'sclr_im'])
    yield from check_shutters(True, 'Close')


# Setup alias/synonym
peakup = smart_peakup
# peakup = doms_peakup


def plot_all_peakup(scanid=-1):
    def normalize_y(d, norm_min=None, norm_max=None):
        if norm_min is None:
            norm_min = np.amin(d)
        if norm_max is None:
            norm_max = np.amax(d)
        return (d - norm_min) / (norm_max - norm_min)

    bs_run = c[int(scanid)]
    ds = bs_run['primary']['data']
    ds_keys = list(ds.keys())
    fig, ax = plt.subplots()
    x = ds['dcm_c2_pitch']
    arg_sort = np.argsort(x)
    if 'xbpm1_sumT' in ds_keys:
        ax.plot(x[arg_sort], normalize_y(ds['xbpm1_sumT'])[arg_sort], label='XBPM-1')
    if 'bpm4_total_current' in ds_keys:
        ax.plot(x[arg_sort],
                normalize_y(ds['bpm4_total_current'][:], norm_min=0.0002)[arg_sort],
                label='B-hutch XBPM')
    if 'bpm5_total_current' in ds_keys:
        ax.plot(x[arg_sort], normalize_y(ds['bpm5_total_current'])[arg_sort], label='B-hutch SSA')
    if 'xbpm2_sumT' in ds_keys:
        ax.plot(x[arg_sort],
                normalize_y(ds['xbpm2_sumT'][:], norm_min=0.024)[arg_sort],
                label='XBPM-2')
    if 'xbpm2_total_current' in ds_keys:
        ax.plot(x[arg_sort],
                normalize_y(ds['xbpm2_total_current'][:], norm_min=-0.01531)[arg_sort],
                label='XBPM-2')
    if 'sclr_i0' in ds_keys:
        ax.plot(x[arg_sort], normalize_y(ds['sclr_i0'])[arg_sort], label='I0')
    ax.set_xlabel('DCM C2 Pitch')
    ax.set_ylabel('Normalized Counts')
    ax.legend()

def ic_energy_batch(estart, estop, npts,
                    acqtime=1.0, count_pts=50, outfile=None):
    if (outfile is not None):
        outfile = '%s_SRXionchamber_readings.csv' % (datetime.datetime.now().strftime('%Y%m%d'))
        outdir = '/home/xf05id1/current_user_data/'
        ion_chamber_fp = open(outdir + outfile, 'w')
        ion_chamber_fp.write('# Energy, ICM, IC0, ICT\n')
    try:
        # Setup scaler and open shutter
        yield from abs_set(sclr1.preset_time, acqtime)
        yield from bps.mov(shut_b, 'Open')

        for i in np.linspace(estart, estop, num=npts):
            yield from mv(energy, i)
            yield from bps.sleep(10)
            yield from peakup_fine(shutter=False)
            yield from bps.sleep(10)
            yield from count([sclr1], num=count_pts)

            if (outfile is not None):
                tbl = db[-1].table()
                icm_mean = tbl['sclr_im'].mean
                ic0_mean = tbl['sclr_i0'].mean
                ict_mean = tbl['sclr_it'].mean
                ion_chamber_fp.write('%8.0f, %d, %d, %d\n' % (i, icm_mean, ic0_mean, ict_mean))

        # Close the shutter
        yield from bps.mov(shut_b, 'Close')
    finally:
        if (outfile is not None):
            ion_chamber_fp.close()


def hdcm_bragg_temperature(erange, estep, dwell, N, dt=0):
    # Loop to test the bragg temperature during a XANES scan

    # Convert erange and estep to numpy array
    ept = np.array([])
    erange = np.array(erange)
    estep = np.array(estep)
    # Calculation for the energy points
    for i in range(len(estep)):
        ept = np.append(ept, np.arange(erange[i], erange[i+1], estep[i]))
    ept = np.append(ept, np.array(erange[-1]))

    dets = [dcm.temp_pitch, energy.energy, dcm.bragg, dcm.c1_roll, dcm.c2_pitch, bpm4]
    dets_by_name = [d.name
                    for d in dets]
    if ('bpm4' in dets_by_name):
        dets_by_name.append('bpm4_x')
        dets_by_name.append('bpm4_y')

    livecallbacks = LiveTable(dets_by_name)

    def custom_perstep(detectors, motor, step):
        def move():
            grp = _short_uid('set')
            yield Msg('checkpoint')
            yield Msg('set', motor, step, group=grp)
            yield Msg('wait', None, group=grp)

        yield from move()
        yield from bps.sleep(dwell)
        yield from trigger_and_read(list(detectors) + [motor])

    @subs_decorator({'all' : livecallbacks})
    def myscan():
        yield from list_scan(dets, energy, list(ept), per_step=custom_perstep)

    for i in range(N):
        yield from myscan()
        yield from bps.sleep(dt)

    return


# braggcalib(use_xrf=True)
# ------------------------------------------------------------------- #
"""
Created on Wed Jun 17 17:03:46 2015
new script for doing HDCM C1Roll and C2X calibration automatically
it requires the HDCM Bragg is calibrated and the d111 and dBragg in SRXenergy script are up-to-date

converting to be compatible with bluesky, still editing
"""
# import SRXenergy  # Not used in this file
import time  # time.sleep should be changed to bps.sleep if used in plan
import string
from matplotlib import pyplot
import subprocess
import scipy as sp
import scipy.optimize
import math
import numpy as np
# import srxbpm  # Not used in this file

def hdcm_c1roll_c2x_calibration():
    onlyplot = False
    #startTi = False
    usecamera = True
    endstation = False

    numAvg = 10


    print(energy._d_111)
    print(energy._delta_bragg)

    if endstation == False:  #default BPM1
        q=38690.42-36449.42 #distance of observing point to DCM; here observing at BPM1
        camPixel=0.006 #mm
        # expotimePV = 'XF:05IDA-BI:1{BPM:1-Cam:1}AcquireTime'
        expotimePV = bpmAD.cam.acquire_time
    else:
        q=(62487.5+280)-36449.42 #distance of observing point to DCM; here observing at 28 cm downstream of M3. M3 is at 62.4875m from source
        camPixel=0.00121 #mm
        # expotimePV = 'XF:05IDD-BI:1{Mscp:1-Cam:1}AcquireTime'
        expotimePV = hfvlmAD.cam.acquire_time



    if onlyplot == False:

        if endstation == True:
            # cenxPV= 'XF:05IDD-BI:1{Mscp:1-Cam:1}Stats1:CentroidX_RBV'
            # cenyPV= 'XF:05IDD-BI:1{Mscp:1-Cam:1}Stats1:CentroidY_RBV'
            cenxPV = hfvlmAD.stats1.centroid.x
            cenyPV = hfvlmAD.stats1.centroid.y
        else:
            # cenxPV= 'XF:05IDA-BI:1{BPM:1-Cam:1}Stats1:CentroidX_RBV'
            # cenyPV= 'XF:05IDA-BI:1{BPM:1-Cam:1}Stats1:CentroidY_RBV'
            cenxPV = bpmAD.stats1.centroid.x
            cenyPV = bpmAD.stats1.centroid.y

        # bragg_rbv = PV('XF:05IDA-OP:1{Mono:HDCM-Ax:P}Mtr.RBV')
        # bragg_val = PV('XF:05IDA-OP:1{Mono:HDCM-Ax:P}Mtr.VAL')
        bragg_rbv = dcm.bragg.user_readback
        bragg_val = dcm.bragg.user_setpoint


        # ctmax = PV('XF:05IDA-BI:1{BPM:1-Cam:1}Stats1:MaxValue_RBV')
        ctmax = bpmAD.stats1.max_value
        # expo_time = PV('XF:05IDA-BI:1{BPM:1-Cam:1}AcquireTime_RBV')
        expo_time = bpmAD.cam.acquire_time

        # PV not found
        # only usage is commented out
        # umot_go = PV('SR:C5-ID:G1{IVU21:1-Mtr:2}Sw:Go')

        #know which edges to go to
        #if startTi == True:
        #    elementList=['Ti', 'Fe', 'Cu', 'Se']
        #else:
        #    elementList=['Se', 'Cu', 'Fe', 'Ti']

        if endstation == False:
            #if dcm_bragg.position > 15:
            if bragg_rbv.get() > 15:
                #elementList=['Ti', 'Cr', 'Fe', 'Cu', 'Se']
                #Ti requires exposure times that would require resetting the
                #threshold in the stats record
                elementList=['Cr', 'Fe', 'Cu', 'Se']
            else:
                #elementList=['Se', 'Cu', 'Fe', 'Cr', 'Ti']
                elementList=['Se', 'Cu', 'Fe', 'Cr']
        else:
            if bragg_rbv.get() > 13:
            #if dcm_bragg.position > 13:
                elementList=['Ti', 'Cr', 'Fe', 'Cu', 'Se']
            else:
                    elementList=['Se', 'Cu', 'Fe', 'Cr', 'Ti']


        energyDic={'Cu':8.979, 'Se': 12.658, 'Fe':7.112, 'Ti':4.966, 'Cr':5.989}
        harmonicDic={'Cu':5, 'Se': 5, 'Fe':3, 'Ti':3, 'Cr':3}            #150 mA, 20151007

        #use for camera option
        expotime={'Cu':0.003, 'Fe':0.004, 'Se':0.005, 'Ti':0.015, 'Cr':0.006}  #250 mA, 20161118, BPM1
        #expotime={'Cu':0.005, 'Fe':0.008, 'Se':0.01, 'Ti':0.03, 'Cr':0.0012}  #150 mA, 20151110, BPM1
        #expotime={'Cu':0.1, 'Fe':0.2, 'Se':0.2, 'Cr': 0.3}  #150 mA, 20151007, end-station

        #use for bpm option
        foilDic={'Cu':25.0, 'Se': 0.0, 'Fe':25.0, 'Ti':25}

        centroidX={}
        centroidY={}

        theoryBragg=[]
        dx=[]
        dy=[]


        C2Xval = dcm.c2_x.user_setpoint.get()
        C1Rval = dcm.c1_roll.user_setpoint.get()


        #dBragg=SRXenergy.whdBragg()
        dBragg = energy._delta_bragg

        for element in elementList:
            centroidXSample=[]
            centroidYSample=[]

            print(element)
            E=energyDic[element]
            print('Edge:', E)


            energy.move_c2_x.put(False)
            energy.move(E,wait=True)
#            energy.set(E)
#
#            while abs(energy.energy.position - E) > 0.001 :
#                time.sleep(1)

            print('done moving energy')
            #BraggRBV, C2X, ugap=SRXenergy.EtoAll(E, harmonic = harmonicDic[element])

            #print BraggRBV
            #print ugap
            #print C2X, '\n'

            #go to the edge

            #ugap_set=PV('SR:C5-ID:G1{IVU21:1-Mtr:2}Inp:Pos')
            #ugap_rbv=PV('SR:C5-ID:G1{IVU21:1-LEnc}Gap')

#            print 'move undulator gap to:', ugap
            #ivu1_gap.move(ugap)
#            ugap_set.put(ugap, wait=True)
#            umot_go.put(0)
#            time.sleep(10)

#            while (ugap_rbv.get() - ugap) >=0.01 :
#                time.sleep(5)
#            time.sleep(2)

#            print 'move Bragg to:', BraggRBV
#            bragg_val.put(BraggRBV, wait= True)
#            while (bragg_rbv.get() - BraggRBV) >=0.01 :
#                time.sleep(5)
            #dcm_bragg.move(BraggRBV)
#            time.sleep(2)

            if usecamera == True:
                expotimePV.put(expotime[element])
                while ctmax.get() <= 200:
                    expotimePV.put(expo_time.get() + 0.001)
                    print('increasing exposuring time.')
                    time.sleep(0.6)
                while ctmax.get() >= 180:
                    expotimePV.put(expo_time.get() - 0.001)
                    print('decreasing exposuring time.')
                    time.sleep(0.6)
                print('final exposure time =' + str(expo_time.get()))
                print('final max count =' + str(ctmax.get()))


                #record the centroids on BPM1 camera
                print('collecting positions with', numAvg, 'averaging...')
                for i in range(numAvg):
                    centroidXSample.append(cenxPV.get())
                    centroidYSample.append(cenyPV.get())
                    time.sleep(2)
                if endstation == False:
                    centroidX[element] = sum(centroidXSample)/len(centroidXSample)
                else:
                    #centroidX[element] = 2452-sum(centroidXSample)/len(centroidXSample)
                    centroidX[element] = sum(centroidXSample)/len(centroidXSample)

                centroidY[element] = sum(centroidYSample)/len(centroidYSample)

                print(centroidXSample)
                print(centroidYSample)
                #print centroidX, centroidY

                dx.append(centroidX[element]*camPixel)
                dy.append(centroidY[element]*camPixel)

                print(centroidX)
                print(centroidY, '\n')
            #raw_input("press enter to continue...")

    #        else:
    #
    #            bpm1_y.move(foilDic[element])
    #            time.sleep(2)
    #            position=bpm1.Pavg(Nsamp=numAvg)
    #            dx.append(position['H'])
    #            dy.append(position['V'])
    #            print dx
    #            print dy

            theoryBragg.append(energy.bragg.position+dBragg)

        #fitting

            #fit centroid x to determine C1roll
            #fit centroid y to determine C2X


        if endstation == True:
            temp=dx
            dx=dy
            dy=temp

        print('C2Xval=', C2Xval)
        print('C1Rval=', C1Rval)
        print('dx=', dx)
        print('dy=', dy)
        print('theoryBragg=', theoryBragg)

    else:
        # Is this different from definition above?
        C2Xval = dcm.c2_x.user_setpoint.get()
        C1Rval = dcm.c1_roll.user_setpoint.get()

    fitfunc = lambda pa, x: pa[1]*x+pa[0]
    errfunc = lambda pa, x, y: fitfunc(pa,x) - y

    pi=math.pi
    sinBragg=np.sin(np.array(theoryBragg)*pi/180)
    sin2Bragg=np.sin(np.array(theoryBragg)*2*pi/180)
    print('sinBragg=', sinBragg)
    print('sin2Bragg=', sin2Bragg)


    guess = [dx[0], (dx[-1]-dx[0])/(sinBragg[-1]-sinBragg[0])]
    fitted_dx, success = sp.optimize.leastsq(errfunc, guess, args = (sinBragg, dx))
    print('dx=', fitted_dx[1], '*singBragg +', fitted_dx[0])

    droll=fitted_dx[1]/2/q*1000 #in mrad
    print('current C1Roll:', C1Rval)
    print('current C1Roll is off:', -droll)
    print('calibrated C1Roll:', C1Rval + droll, '\n')

    sin2divBragg = sin2Bragg/sinBragg
    print('sin2divBragg=', sin2divBragg)


    guess = [dy[0], (dy[-1]-dy[0])/(sin2divBragg[-1]-sin2divBragg[0])]
    fitted_dy, success = sp.optimize.leastsq(errfunc, guess, args = (sin2divBragg, dy))
    print('dy=', fitted_dy[1], '*(sin2Bragg/sinBragg) +', fitted_dy[0])
    print('current C2X:', C2Xval)
    print('current C2X corresponds to crystal gap:', fitted_dy[1])

    pyplot.figure(1)
    pyplot.plot(sinBragg, dx, 'b+')
    pyplot.plot(sinBragg, sinBragg*fitted_dx[1]+fitted_dx[0], 'k-')
    pyplot.title('C1Roll calibration')
    pyplot.xlabel('sin(Bragg)')
    if endstation == False:
        pyplot.ylabel('dx at BPM1 (mm)')
    else:
        pyplot.ylabel('dx at endstation (mm)')
    pyplot.show()

    pyplot.figure(2)
    pyplot.plot(sin2divBragg, dy, 'b+')
    pyplot.plot(sin2divBragg, sin2divBragg*fitted_dy[1]+fitted_dy[0], 'k-')
    pyplot.title('C2X calibration')
    pyplot.xlabel('sin(2*Bragg)/sin(Bragg)')
    if endstation == False:
        pyplot.ylabel('dy at BPM1 (mm)')
    else:
        pyplot.ylabel('dy at endstation (mm)')
    pyplot.show()
