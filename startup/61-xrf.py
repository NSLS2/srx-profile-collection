print(f'Loading {__file__}...')


import epics
import os
import collections
import numpy as np
import time as ttime
import matplotlib.pyplot as plt

import bluesky.plans as bp
from bluesky.plans import outer_product_scan, scan
from bluesky.callbacks import LiveGrid
from bluesky.callbacks.fitting import PeakStats
from bluesky.preprocessors import subs_wrapper
import bluesky.plan_stubs as bps
from bluesky.plan_stubs import mv, abs_set
from bluesky.simulators import plot_raster_path


# def fermat_plan(x_range, y_range, dr, factor, exp_time=0.2):
def fermat_plan(*args, **kwargs):
    x_range = args[0]
    y_range = args[1]
    dr = args[2]
    factor = args[3]
    # print(f'{x_range}\t{y_range}')
    # print(args)
    kwargs.setdefault('exp_time', 0.2)

    # Setup motors
    x_motor = nano_stage.sx
    y_motor = nano_stage.sy

    # Setup detectors
    dets = [sclr1, xs, xbpm2, merlin, bpm4, temp_nanoKB]

    # print("ready to call fermat_master...")
    yield from fermat_master_plan(dets, x_motor, y_motor, *args, **kwargs)


def fermat_master_plan(*args, exp_time=None, **kwargs):
    # Synchronize exposure times
    sclr1.preset_time.put(exp_time)
    xs.external_trig.put(False)
    xs.cam.acquire_time.put(exp_time)
    merlin.cam.acquire_time.put(exp_time)
    merlin.cam.acquire_period.put(exp_time + 0.005)

    scan_md = {}
    get_stock_md(scan_md)
    scan_md['scan']['merlin'] = {'merlin_exp_time' : exp_time,
                                 'merlin_exp_period' : exp_time + 0.005}

    plan = bp.rel_spiral_fermat(*args, **kwargs)
    d = plot_raster_path(plan, args[1].name, args[2].name, probe_size=.001, lw=0.5)
    num_points = d['path'].get_path().vertices.shape[0]

    print(f"Number of points: {num_points}")
    xs.total_points.put(num_points)
    yield from bps.mv(merlin.total_points, num_points,
                      merlin.hdf5.num_capture, num_points)
    merlin.hdf5.stage_sigs['num_capture'] = num_points
    yield from rel_spiral_fermat(*args, **kwargs, md=scan_md)


# Check the fermat spiral points
# This does not run within the run engine
# plot_raster_path(rel_spiral_fermat([], nano_stage.sx, nano_stage.sy, 2, 2,
# 0.5, 1), nano_stage.sx.name, nano_stage.sy.name)
def check_fermat_plan(xrange, yrange, dr, factor):
    xmotor = nano_stage.sx
    ymotor = nano_stage.sy
    plot_raster_path(rel_spiral_fermat([], xmotor, ymotor, xrange, yrange, dr, factor), xmotor.name, ymotor.name)
    ax = plt.gca()
    line = ax.lines[0]
    print(f'The scan will have {len(line.get_xdata())} points.')


import h5py
from skimage.io import imsave

def export_merlin_from_tiled(scanid=-1, wd=None):
    if wd is None:
        wd = "."

    # Get all documents for a run
    run = c[int(scanid)]
    scanid = int(run.start["scan_id"])

    docs = [doc for doc in run.documents() if "resource" in doc[0]]
    merlin_docs = [doc for doc in docs if "MERLIN" in doc[1]["spec"]]
    merlin_files = [doc[1]["resource_path"] for doc in merlin_docs]

    N_files = len(merlin_files)
    idx = 0
    for fn in merlin_files:
        print(f"Opening file {fn.split('/')[-1]} ({(idx // N_files)+1:3d}/{N_files:3d})...")
        with h5py.File(fn, 'r') as f:
            d = f["entry/data/data"]
            N, _, _ = d.shape
            for i in range(N):
                # print(f"  Writing frame ({i+1:3d}/{N:3d})...")
                img = d[i, :, :]
                img_fn = os.path.join(wd, f"scan{scanid}_{idx:08d}.tif")
                imsave(img_fn, img)
                idx += 1


def export_flying_merlin2tiff(scanid=-1, wd=None):
    if wd is None:
        wd = '/home/xf05id1/current_user_data/'

    print('Loading data...')
    h = db[int(scanid)]
    d = h.data('merlin_image', stream_name='stream0', fill=True)
    d = np.array(list(d))
    d = np.squeeze(d)
    d = np.array(d, dtype='float32')
    x = np.array(list(h.data('enc1', stream_name='stream0', fill=True)))
    y = np.array(list(h.data('enc2', stream_name='stream0', fill=True)))
    I0= np.array(list(h.data('i0', stream_name='stream0', fill=True)))

    # Flatten arrays
    (N, M) = x.shape
    x_flat = np.reshape(x, (N*M, ))
    y_flat = np.reshape(y, (N*M, ))
    I0_flat = np.reshape(I0, (N*M, ))

    # Get scanid
    if (scanid < 0):
        scanid = h.start['scan_id']

    print('Writing data...')
    fn = 'scan%d.tif' % scanid
    fn_txt = 'scan%d.txt' % scanid
    io.imsave(wd + fn, d)
    np.savetxt(wd + fn_txt, np.array((x_flat, y_flat, I0_flat)))
       # HACK to make sure we clear the cache.  The cache size is 1024 so
    # this would eventually clear, however on this system the maximum
    # number of open files is 1024 so we fail from resource exaustion before
    # we evict anything.
    db._catalog._entries.cache_clear()
    gc.collect()


def export_merlin2tiff(scanid=-1, wd=None):
    if wd is None:
        wd = '/home/xf05id1/current_user_data/'

    print('Loading data...')
    h = db[int(scanid)]
    d = h.data('merlin_image', fill=True)
    d = np.array(list(d))
    d = np.squeeze(d)
    d = np.array(d, dtype='float32')
    x = np.array(list(h.data('nano_stage_sx', fill=True)))
    y = np.array(list(h.data('nano_stage_sy', fill=True)))
    I0= np.array(list(h.data('sclr_i0', fill=True)))

    # Get scanid
    if (scanid < 0):
        scanid = h.start['scan_id']

    print('Writing data...')
    fn = 'scan%d.tif' % scanid
    fn_txt = 'scan%d.txt' % scanid
    io.imsave(wd + fn, d)
    np.savetxt(wd + fn_txt, np.array((x, y, I0)))
    # HACK to make sure we clear the cache.  The cache size is 1024 so
    # this would eventually clear, however on this system the maximum
    # number of open files is 1024 so we fail from resource exaustion before
    # we evict anything.
    db._catalog._entries.cache_clear()
    gc.collect()


@parameter_annotation_decorator({
    "parameters": {
        "xmotor": {"default": "'hf_stage.sx'"},
        "ymotor": {"default": "'hf_stage.sy'"},
    }
})
def nano_xrf(xstart, xstop, xnum,
             ystart, ystop, ynum, dwell,
             shutter=True, extra_dets=None,
             xmotor=nano_stage.sx, ymotor=nano_stage.sy, snake=True,
             vlm_snapshot=False, snapshot_after=False, N_dark=10):
    
    # Check for negative number of points
    if (xnum < 1 or ynum < 1):
        raise ValueError('Number of points must be positive!')

    # Setup detectors
    if extra_dets is None:
        extra_dets = []
    dets = [sclr1, xs, xmotor, ymotor] + extra_dets

    # Record relevant metadata in the Start document, defined in 90-usersetup.py
    scan_md = {}
    get_stock_md(scan_md)
    scan_md['scan']['type'] = 'XRF_STEP'
    scan_md['scan']['scan_input'] = [xstart, xstop, xnum, ystart, ystop, ynum, dwell]
    # scan_md['scan']['detectors'] = [d.name for d in dets]
    scan_md['dwell'] = dwell
    scan_md['scan']['fast_axis'] = {'motor_name' : xmotor.name,
                                    'units' : xmotor.motor_egu.get()}
    scan_md['scan']['slow_axis'] = {'motor_name' : ymotor.name,
                                    'units' : ymotor.motor_egu.get()}
    scan_md['scan']['theta'] = {'val' : np.round(nano_stage.th.user_readback.get(), decimals=3),
                                'units' : nano_stage.th.motor_egu.get()}
    scan_md['scan']['delta'] = {'val' : 0,
                                'units' : xmotor.motor_egu.get()}
    scan_md['scan']['snake'] = 1 if snake else 0
    scan_md['scan']['shape'] = (xnum, ynum)
    md_dets = list(dets)
    if vlm_snapshot:
        md_dets = md_dets + [nano_vlm]
    get_det_md(scan_md, md_dets)

    # Set xs mode to step.
    xs.mode = SRXMode.step

    # Set counting time
    sclr1.preset_time.put(dwell)
    xs.cam.acquire_time.put(dwell)
    xs.total_points.put(xnum * ynum)
    if (merlin in dets):
        merlin.cam.acquire_time.put(dwell)
        merlin.cam.acquire_period.put(dwell + 0.005)
        merlin.hdf5.stage_sigs['num_capture'] = xnum * ynum
        scan_md['scan']['merlin'] = {'merlin_exp_time' : dwell,
                                     'merlin_exp_period' : dwell + 0.005}

    # LiveTable
    livecallbacks = []

    # roi_key = getattr(xs.channel1.rois, roi_name).value.name
    roi_key = xs.channel01.get_mcaroi(mcaroi_number=1).total_rbv.name
    livecallbacks.append(LiveTable([xmotor.name, ymotor.name, roi_key]))
    # livetableitem.append(roi_key)
    # roi_name = 'roi{:02}'.format(1)

    livecallbacks.append(LiveGrid((ynum, xnum), roi_key,
                                  clim=None, cmap='viridis',
                                  xlabel='x [um]', ylabel='y [um]',
                                  extent=[xstart, xstop, ystart, ystop],
                                  x_positive='right', y_positive='down'))
    
    @run_decorator(md=scan_md)
    @vlm_decorator(vlm_snapshot, after=snapshot_after)
    @dark_decorator(dets, N_dark=N_dark, shutter=shutter)
    def plan():
        yield from mod_grid_scan(dets,
                                 ymotor, ystart, ystop, ynum,
                                 xmotor, xstart, xstop, xnum, snake,
                                 run_agnostic=True)
    myplan = plan() # This line feels silly

    # myplan = grid_scan(dets,
    #                    ymotor, ystart, ystop, ynum,
    #                    xmotor, xstart, xstop, xnum, snake,
    #                    md=scan_md)

    myplan = subs_wrapper(myplan,
                          {'all': livecallbacks})

    # Open shutter
    yield from check_shutters(shutter, 'Open')

    # grid scan
    uid = yield from myplan

    # Close shutter
    yield from check_shutters(shutter, 'Close')

    return uid