print(f'Loading {__file__}...')


import sys
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

from ophyd.areadetector.plugins import (ImagePlugin_V33, TIFFPlugin_V33,
                                        ROIPlugin_V33, StatsPlugin_V33)


class SRXTIFFPlugin(TIFFPlugin,
                    FileStoreTIFF,
                    FileStoreIterativeWrite):
    ...


class SRXAreaDetectorCam(AreaDetectorCam):
    pool_max_buffers = None


class SRXCamera(SingleTrigger, AreaDetector):
    def __init__(self, *args, root_path='/nsls2/data/srx/assets', **kwargs):
        super().__init__(*args, **kwargs)
        self.read_attrs = ['tiff', 'stats5']
        self.stats5.read_attrs = ['total']
        self.tiff.write_path_template=f'{root_path}/{self.name}/%Y/%m/%d/'
        self.tiff.read_path_template=f'{root_path}/{self.name}/%Y/%m/%d/'
        self.tiff.reg_root=f'{root_path}/{self.name}'

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
nano_vlm = create_camera('XF:05ID1-ES{PG-Cam:1}', 'nano_vlm', root_path='/nsls2/data/srx/legacy')
# hfvlm_AD = create_camera('XF:05IDD-BI:1{Mscp:1-Cam:1}', 'hfvlmAD', root_path='/nsls2/data/srx/legacy')
camd01 = create_camera('XF:05IDD-BI:1{Mscp:1-Cam:1}', 'camd01', root_path='/nsls2/data/srx/legacy')
if camd01 is not None:
    camd01.read_attrs = ['tiff', 'stats1', 'stats2', 'stats3', 'stats4']
    camd01.tiff.read_attrs = []
# camd05 = create_camera('XF:05IDD-BI:1{Cam:5}', 'camd05', root_path='/nsls2/data/srx/legacy')
camd06 = create_camera('XF:05IDD-BI:1{Cam:6}', 'camd06', root_path='/nsls2/data/srx/legacy')
camd08 = create_camera('XF:05IDD-BI:1{Cam:8}', 'camd08', root_path='/nsls2/data/srx/legacy')
