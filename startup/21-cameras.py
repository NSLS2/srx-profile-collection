print(f'Loading {__file__}...')


import sys
import functools
import datetime
from ophyd import EpicsSignal
from ophyd.areadetector import (AreaDetector, ImagePlugin,
                                TIFFPlugin, StatsPlugin,
                                ROIPlugin, TransformPlugin,
                                OverlayPlugin, ProcessPlugin)
from ophyd.areadetector.plugins import PvaPlugin
from ophyd.areadetector.filestore_mixins import (FileStoreIterativeWrite,
                                                 FileStoreTIFF)
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


class SRXAreaDetectorCam(AreaDetectorCam):
    pool_max_buffers = None


class SRXCamera(SingleTrigger, AreaDetector):


    def __init__(self, *args, root_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_attrs = ['tiff', 'stats5']
        self.stats5.read_attrs = ['total']
        
        if root_path is None: # Post data security
            self.tiff.write_path_template=self.path_start + self.path_template_str(self.root_path_str())
            self.tiff.read_path_template=self.path_start + self.path_template_str(self.root_path_str())
            self.tiff.reg_root=self.path_start + self.root_path_str()
        else: # Pre data security
            self.tiff.write_path_template=f'{root_path}/{self.name}/%Y/%m/%d/'
            self.tiff.read_path_template=f'{root_path}/{self.name}/%Y/%m/%d/'
            self.tiff.reg_root=f'{root_path}/{self.name}'

    path_start = "/nsls2/data/srx/"

    def root_path_str(self):
        data_session = RE.md["data_session"]
        cycle = RE.md["cycle"]
        if "Commissioning" in get_proposal_type():
            root_path = f"proposals/commissioning/{data_session}/assets/{self.name}/"
        else:
            root_path = f"proposals/{cycle}/{data_session}/assets/{self.name}/"
        return root_path

    def path_template_str(self, root_path):
        path_template = "%Y/%m/%d/"
        return root_path + path_template

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
    tiff = Cpt(SRXTIFFPlugin, 'TIFF1:',
               write_path_template='%Y/%m/%d/',
               read_path_template='%Y/%m/%d/',
               # root=root_path)
               )

def create_camera(pv, name, root_path='/nsls2/data/srx/assets'):
    try:
        cam = SRXCamera(pv, name=name, root_path=root_path)
    except TimeoutError:
        print(f'\nCannot connect to {name}. Continuing without device.\n')
        cam = None
    except Exception as ex:
        print(ex, end='\n\n')
        cam = None
    return cam


hfm_cam = create_camera('XF:05IDA-BI:1{FS:1-Cam:1}', 'hfm_cam')
bpmA_cam = create_camera('XF:05IDA-BI:1{BPM:1-Cam:1}', 'bpmA_cam')
# nano_vlm = create_camera('XF:05ID1-ES{PG-Cam:1}', 'nano_vlm', root_path='/nsls2/data/srx/legacy')
nano_vlm = create_camera('XF:05ID1-ES{PG-Cam:1}', 'nano_vlm', root_path=None)
# hfvlm_AD = create_camera('XF:05IDD-BI:1{Mscp:1-Cam:1}', 'hfvlmAD', root_path='/nsls2/data/srx/legacy')
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
            if vlm_snapshot:
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