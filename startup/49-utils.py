print(f'Loading {__file__}...')


import os
import lmfit
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf
from bluesky.callbacks import LiveFit, LiveFitPlot
from bluesky.callbacks.fitting import PeakStats
from bluesky.plan_stubs import mv
from bluesky.utils import short_uid
from bluesky.plan_stubs import checkpoint, abs_set, wait, trigger_and_read


# Simple Gaussian
def gaussian(x, A, sigma, x0):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2))


# More complete Gaussian with offset and slope
def f_gauss(x, A, sigma, x0, y0, m):
    return y0 + m * x + A * np.exp(-(x - x0)**2 / (2 * sigma**2))


# Integral of the Gaussian function with slope and offset
def f_int_gauss(x, A, sigma, x0, y0, m):
    x_star = (x - x0) / sigma
    return A * erf(x_star / np.sqrt(2)) + y0 + m * x


# Error function with offset
def f_offset_erf(x, A, sigma, x0, y0):
    x_star = (x - x0) / sigma
    return A * erf(x_star / np.sqrt(2)) + y0


# Let's fit two error functions
def f_two_erfs(x, A1, sigma1, x1, y1,
                  A2, sigma2, x2, y2):
    f_combo = (f_offset_erf(x, A1, sigma1, x1, y1) +
               f_offset_erf(x, A2, sigma2, x2, y2))
    return f_combo


def mv_position(pos=[]):
    """
    Move to predefined positions of diving board(pos1,default),
    the schitillator(pos2), or the Cu wire(pos3).

    pos     <list> 1 = [4682, 3365.5, -7201.7] # diving board
                   2 = [32.11,17.646,61.374] # scintillator
                   3 = [[30.114,22.464,53.874] # ANT Siemens star
                   [x, y, z] # any positions defined
    """
    print('To go to position 1,2,3; '
          'Assuming topx, topz at 0; rotation at 0 deg.')

    # Check current positions
    if (pos == []):
        print(f"You are now in this position: "
              "{nano_stage.x.position}, "
              "{nano_stage.y.position}, "
              "{nano_stage.z.position}")
        print('No new positions given. Exiting...')
        return

    # Check positions and go there
    if (pos == 1):
        print('Will go to diving board position.')
        pos = [4682, 3365.5, -7201.7]
    elif (pos == 2):
        print('Will go to scintillator position.')
        pos = [32.11, 17.064, 61.454]
    elif (pos == 3):
        print('Will go to Simens Star position.')
        pos = [30.114, 22.464, 53.874]
    elif (len(pos)>1):
        print(f'You will move to the defined positions: {pos} now.')
        if len(pos) <3:
            yield from mv(nano_stage.x, pos[0],
                          nano_stage.y, pos[1])
        else:
            yield from mv(nano_stage.x, pos[0],
                          nano_stage.y, pos[1],
                          nano_stage.z, pos[2])
    else:
        print('Not a position, exiting...')
        return

def copyscanparam(src_num, dest_num):
    '''
    Copy all scan paramaters from scan src_num to scan dest_num
    wrapper for cp method in python scanrecord object
    '''
    src = 'scan{}'.format(src_num-1)
    dest = 'scan{}'.format(dest_num-1)
    scanrecord.cp(src, dest)


def printfig():
    plt.savefig('/home/xf05id1/tmp/temp.png',
                bbox_inches='tight',
                pad_inches=4)
    os.system("lp -d HXN-printer-1 /home/xf05id1/tmp/temp.png")


def print_warning_message(msg):
    msg_len = len(msg) + 2
    print(f"\n{'*' * msg_len}")
    print(f' {msg} ')
    print(f"{'*' * msg_len}\n")


def banner(str_list, border="-"):
    if not isinstance(str_list, list):
        str_list = [str_list]

    N = 2
    for str in str_list:
        N = max(len(str), N)

    print(border * (N + 2))
    for str in str_list:
        print(f" {str}")
    print(border * (N + 2), end='\n\n')


def print_baseline(scanid=-1, key_filter=None):
    '''
    Print all the baseline metadata.

    Input
    -----
    scanid : int
      the scan ID for the scan of interest.

    Returns
    -------
    Nothing
    '''

    scanid = int(scanid)
    h = db[scanid]
    tbl = h.table('baseline')
    pd.set_option('display.max_rows', 999)
    if (key_filter is not None):
        all_keys = tbl.keys()
        filtered_list = []
        for key in all_keys:
            if key_filter in key:
                filtered_list.append(key)
        tbl = tbl[filtered_list]
    print(tbl.T)
    pd.reset_option('display.max_rows')


def estimate_scan_duration(fastaxis_num, slowaxis_num, dwell, scantype='XRF_FLY', event_delay=None):
    '''
    xnum    int     number of points as entered for the scan in X
    ynum    int     number of points as entered for the scan in Y
    dwell   float   exposure time in seconds as entered on the command line
    scantype    string  one of [XRF, XRF_fly, XANES]
    '''
    #overhead = {'xrf': 0.7, 'xrf_fly': 3.8, 'xanes': 1.6}
    overhead = {'xrf': 0.7, 'XRF_FLY': 5.5, 'xanes': 1.6}
    if event_delay is None:
        try:
            delay = overhead[scantype.casefold()]
        except KeyError:
            print("Warning: scantype is not supported, delay = 0s")
            delay = 0.
    else:
        delay = event_delay
        print(f"overhead per line is {delay}s")

##    if scantype.casefold() == 'xrf_fly':
##        if delay != 0.:
##            delay = delay / xnum
##        xnum = xnum - 1
##        ynum = ynum - 1

    #result = ((xnum + 1) * (ynum + 1)) * (dwell + delay)
    '''
    if 
        result = (xnum*ynum) * dwell + ynum*delay
        div, rem = divmod(result, 3600)
        print(f"Estimated duration is {int(div):d} hr {rem / 60:.1f} min "
              "({result:.1f} sec).")
    '''
    return result


def custom_one_nd_step(detectors, step, pos_cache):
    """
    Inner loop of an N-dimensional step scan

    This is the default function for ``per_step`` param in ND plans.

    Parameters
    ----------
    detectors : iterable
        devices to read
    step : dict
        mapping motors to positions in this step
    pos_cache : dict
        mapping motors to their last-set positions
    """
    def move():
        yield from checkpoint()
        grp = short_uid('set')
        for motor, pos in step.items():
            if pos == pos_cache[motor]:
                # This step does not move this motor.
                continue
            yield from abs_set(motor, pos, group=grp)
            pos_cache[motor] = pos
        yield from wait(group=grp)

    motors = step.keys()
    yield from move()

    # Here is the custom part, add a 1 second delay
    yield from bps.sleep(1)

    yield from trigger_and_read(list(detectors) + list(motors))


def append_kwargs_md(**scan_md_kwargs):
    """
    General decorator for adding keywords to a function that will be appended into the scan metadata.
    """

    def with_metadata(func):
        @functools.wraps(func)
        def wrapped(*args,
                    **kwargs):

            # Because functions may not explicitly have an 'md' keyword argument,
            # there is no way to check if 'md' will be properly handled
            if 'md' in kwargs:
                md = kwargs.pop('md')
                if md is None:
                    md = {}
            else:
                md = {}
            
            if 'scan' not in md:
                md['scan'] = {}
            md['scan'].update(scan_md_kwargs)
            kwargs['md'] = md            

            return func(*args, **kwargs)
        
        return wrapped
    return with_metadata

# Convenience version for adding default kwargs to md
append_srx_kwargs_md = append_kwargs_md(sample_name='')


# TODO: TEST ME!
# Stand-alone vlm snapshot plan
@append_srx_kwargs_md
def vlm_snapshot(md=None):

    # Define metadata
    md = get_stock_md(md)
    md['detectors'] = [nano_vlm.name]
    md['motors'] = []
    md['plan_args'] = {}
    md['plan_name'] = 'vlm_snapshot'
    md['scan']['type'] = 'VLM_SNAPSHOT'
    md['scan']['detectors'] = [nano_vlm.name]

    # Define plan
    @bpp.run_decorator(md=md)
    def plan():
        yield from _camera_snapshot([nano_vlm])

    return (yield from plan())