print(f'Loading {__file__}...')


import sys
import functools
import datetime
from ophyd import EpicsSignal
from ophyd.areadetector import (AreaDetector, ImagePlugin,
                                TIFFPlugin, StatsPlugin,
                                ROIPlugin, TransformPlugin,
                                OverlayPlugin, ProcessPlugin,
                                HDF5Plugin)
from ophyd.areadetector.plugins import PvaPlugin
from ophyd.areadetector.filestore_mixins import (FileStoreIterativeWrite,
                                                 FileStoreTIFF,
                                                 FileStoreHDF5)
from ophyd.areadetector.trigger_mixins import SingleTrigger
from ophyd.areadetector.cam import AreaDetectorCam
from ophyd.device import Component as Cpt
# from ophyd.status import WaitTimeoutError

from ophyd.areadetector.plugins import (ImagePlugin_V33, TIFFPlugin_V33,
                                        ROIPlugin_V33, StatsPlugin_V33)

from bluesky.run_engine import WaitForTimeoutError


class SRXTIFFPlugin(TIFFPlugin,
                    FileStoreTIFF,
                    FileStoreIterativeWrite):
    ...


class SRXHDF5Plugin(HDF5Plugin,
                    FileStoreHDF5,
                    FileStoreIterativeWrite):

    def stage(self, *args, **kwargs):
        _TIMEOUT = 2
        self.capture.set(0, timeout=_TIMEOUT).wait()
        super().stage(*args, **kwargs)


class SRXAreaDetectorCam(AreaDetectorCam):
    pool_max_buffers = None


class SRXCamera(SingleTrigger, AreaDetector):

    path_start = "/nsls2/data/srx/"

    def __init__(self, *args, root_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_attrs = ['tiff', 'stats4', 'stats5', 'hdf5']
        self.stats4.read_attrs = ['total']
        self.stats5.read_attrs = ['total']
        self.root_path = root_path
        # self._mode = SRXMode.fly
        self.set_paths()

    def set_paths(self):
        if self.root_path is None:
            reg_root = f'{self.path_start}{self.root_path_str}'
            full_path = f'{reg_root}{self.path_template_str}'
        else:
            reg_root = f'{self.root_path}/{self.name}/'
            full_path = f'{reg_root}/{self.path_template_str}'

        self.tiff.reg_root = reg_root
        self.tiff.write_path_template = full_path
        self.tiff.read_path_template = full_path

        self.hdf5.reg_root = reg_root
        self.hdf5.write_path_template = full_path
        self.hdf5.read_path_template = full_path


    def stage(self, *args, **kwargs):
        self.set_paths()
        super().stage(*args, **kwargs)


    @property
    def root_path_str(self):
        data_session = RE.md["data_session"]
        cycle = RE.md["cycle"]

        # if "Commissioning" in get_proposal_type():
        #     root_path = f"proposals/commissioning/{data_session}/assets/{self.name}/"
        # else:
        #     root_path = f"proposals/{cycle}/{data_session}/assets/{self.name}/"

        root_path = f"proposals/{cycle}/{data_session}/assets/{self.name}/"

        return root_path

    @property
    def path_template_str(self):
        path_template = "%Y/%m/%d"
        return path_template

    cam = Cpt(AreaDetectorCam, 'cam1:')
    image = Cpt(ImagePlugin, 'image1:')
    pva = Cpt(PvaPlugin, 'Pva1:')  # Not really implemented in ophyd
    proc = Cpt(ProcessPlugin, 'Proc1:')
    over = Cpt(OverlayPlugin, 'Over1:')
    trans = Cpt(TransformPlugin, 'Trans1:')
    roi1 = Cpt(ROIPlugin, 'ROI1:')
    roi2 = Cpt(ROIPlugin, 'ROI2:')
    roi3 = Cpt(ROIPlugin, 'ROI3:')
    roi4 = Cpt(ROIPlugin, 'ROI4:')
    stats1 = Cpt(StatsPlugin, 'Stats1:')
    stats2 = Cpt(StatsPlugin, 'Stats2:')
    stats3 = Cpt(StatsPlugin, 'Stats3:')
    stats4 = Cpt(StatsPlugin, 'Stats4:')
    stats5 = Cpt(StatsPlugin, 'Stats5:')

    total_points = Cpt(Signal,
                       value=1,
                       doc="The total number of points to be taken")
    fly_next = Cpt(Signal,
                   value=False,
                   doc="latch to put the detector in 'fly' mode")

    tiff = Cpt(SRXTIFFPlugin, 'TIFF1:',
               write_path_template=path_start, # Required. Overriden when staging
               read_path_template=None, # Overriden when staging
               )
    hdf5 = Cpt(SRXHDF5Plugin, 'HDF1:',
               write_path_template=path_start,
               read_path_template=None)

def create_camera(pv, name, root_path='/nsls2/data/srx/assets'):
    try:
        cam = SRXCamera(pv, name=name, root_path=root_path)
    except TimeoutError:
        print(f'\nCannot connect to {name}. Continuing without device.\n')
        cam = None
    except Exception as e:
        print(f'\nUnexpected error connecting to {name}:')
        print(f'\t{e.__class__.__name__}:{e}\n')
        cam = None
    return cam


hfm_cam = create_camera('XF:05IDA-BI:1{FS:1-Cam:1}', 'hfm_cam')
bpmA_cam = create_camera('XF:05IDA-BI:1{BPM:1-Cam:1}', 'bpmA_cam')
nano_vlm = create_camera('XF:05ID1-ES{PG-Cam:1}', 'nano_vlm', root_path=None)
camd01 = create_camera('XF:05IDD-BI:1{Mscp:1-Cam:1}', 'camd01', root_path='/nsls2/data/srx/legacy')
if camd01 is not None:
    camd01.read_attrs = ['tiff', 'stats1', 'stats2', 'stats3', 'stats4']
    camd01.tiff.read_attrs = []
# camd05 = create_camera('XF:05IDD-BI:1{Cam:5}', 'camd05', root_path='/nsls2/data/srx/legacy')
# camd06 = create_camera('XF:05IDD-BI:1{Cam:6}', 'camd06', root_path='/nsls2/data/srx/legacy')
camd08 = create_camera('XF:05IDD-BI:1{Cam:8}', 'camd08', root_path='/nsls2/data/srx/legacy')


# Treat like plan stub and use within a run decorator
def _camera_snapshot(cameras=[nano_vlm]):

    if len(cameras) > 0:
        print('Acquiring camera snapshot...')
        # Unstaging as precautionary measure. Likely all cameras are unstaged
        staging_list = [cam._staged == Staged.yes for cam in cameras]
        for staged, cam in zip(staging_list, cameras):
            if staged:
                yield from bps.unstage(cam)
            yield from bps.stage(cam)
        
        # Snapshot!
        # Should even work with different dwell times
        # yield from bps.trigger_and_read(cameras, name='camera_snapshot')
        try:
            yield from mod_trigger_and_read(cameras,
                                            name='camera_snapshot',
                                            timeout=10) # 10 seconds
        except WaitForTimeoutError:
            warn_str = "WARNING: Camera snapshot failed to trigger in designated time. Continuing without..."
            print(warn_str)
        except Exception as e:
            print(e)
            raise(e)

        # Precautionary unstaging
        for staged, cam in zip(staging_list, cameras):
            yield from bps.unstage(cam)
            if staged:
                yield from bps.stage(cam)
        
        # Clear descripter cache
        for cam in cameras:
            yield Msg("clear_describe_cache", cam)


# Decorator version of vlm snapshot. Must happen within open runs.
def vlm_decorator(vlm_snapshot=True, after=True, position=None):
    """
    Decorator to wrap Bluesky plans with a VLM snapshots. This
    decorator must be called within an already open run.

    Parameters
    ----------
    vlm_snapshot : bool, optional
        Flag to enable the VLM snapshots before and after a
        Bluesky plan. True by default.
    after : bool, optional
        Flag to enable the VLM snapshot after a Bluesky plan.
        True by default.
    position : iterable : optional
        Iterable of motors and positions used to align the
        sample for each snapshot. Example: (xmotor, xposition,
        ymotor, yposition, ...). None by default and the 
        snapshot will occur without any moves.

    Example
    -------
    >>> def custom_plan(detector,
    >>>                 vlm_snapshot=True):    

    >>>    @stage_decorator([detector])
    >>>    @run_decorator(md={})
    >>>    @vlm_decorator(vlm_snapshot,
    >>>                   after=True,
    >>>                   position=(high_res_stage.x, 0,
    >>>                             high_res_stage.y, 0))
    >>>    def plan():
    >>>        yield from bps.trigger_and_read(detector)

    >>>    return (yield from plan())

    """
    def inner_decorator(func):
        @functools.wraps(func)
        def func_with_snapshot(*args, **kwargs):
            if vlm_snapshot is True:
                if position is not None:
                    yield from mv(*position)

                yield from _camera_snapshot([nano_vlm])
                yield from func(*args, **kwargs)
                if after:
                    if position is not None:
                        yield from mv(*position)
                    yield from _camera_snapshot([nano_vlm])
            else:
                yield from func(*args, **kwargs)
        
        return func_with_snapshot
    return inner_decorator

try:
    # nano_vlm.cam.acquire.set(0, settle_time=1)
    nano_vlm.cam.image_mode.set(2, settle_time=0.1)
    nano_vlm.cam.trigger_mode.set(0, settle_time=0.1)
    nano_vlm.cam.acquire_time.set(0.099, settle_time=0.1)
    nano_vlm.cam.acquire_period.set(0.1, settle_time=0.1)
    # nano_vlm.cam.acquire.set(1, settle_time=0.1)

    nano_vlm.tiff.stage_sigs['auto_save'] = 'No'
    nano_vlm.hdf5.kind = 'omitted'


except Exception as ex:
    print(ex)


def setup_flying_cameras(dets,
                         dwell,
                         N_images):
    # Convenience function for setting up cameras
    dets_by_name = {d.name : d for d in dets}

    # Setup nano_vlm
    if 'nano_vlm' in dets_by_name:
        det = dets_by_name['nano_vlm']
        det.cam.acquire.set(0)
        det.cam.array_counter.set(0)

        det.cam.stage_sigs['image_mode'] = 2 # continuous
        det.cam.stage_sigs['trigger_mode'] = 1 # bulb

        # det.cam.stage_sigs['acquire_period'] = 0.05
        det.cam.stage_sigs['acquire_time'] = 0.05

        det.cam.stage_sigs['num_images'] = N_images
        # det.stage_sigs['total_points'] = N_images
        det.cam.stage_sigs['array_counter'] = 0

        det.hdf5.kind = 'normal'
        det.hdf5.stage_sigs['num_capture'] = N_images

        det.tiff.kind = 'omitted'
        det.tiff.stage_sigs['enable'] = 0
