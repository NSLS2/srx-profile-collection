print(f'Loading {__file__}...')

import datetime
import itertools
import sys
import numpy as np
from pathlib import PurePath

from ophyd import Signal
from ophyd import Component as Cpt

from ophyd.areadetector import (AreaDetector, PixiradDetectorCam, ImagePlugin,
                                TIFFPlugin, StatsPlugin, HDF5Plugin,
                                ProcessPlugin, ROIPlugin, TransformPlugin,
                                OverlayPlugin, CamBase)
from ophyd.areadetector.base import ADComponent
from ophyd.areadetector.plugins import PluginBase
from ophyd.areadetector.cam import AreaDetectorCam
from ophyd.device import BlueskyInterface
from ophyd.areadetector.trigger_mixins import SingleTrigger
from ophyd.areadetector.filestore_mixins import (FileStoreIterativeWrite,
                                                 FileStoreHDF5IterativeWrite,
                                                 FileStoreTIFFSquashing,
                                                 FileStoreTIFF,
                                                 FileStoreHDF5,
                                                 new_short_uid,
                                                 FileStoreBase,
                                                 FileStorePluginBase,
                                                 )

from hxntools.detectors.merlin import MerlinDetector
# from nslsii.detectors.merlin import MerlinDetector
from hxntools.handlers import register


class BulkMerlin(BulkXspress):
    HANDLER_NAME = 'MERLIN_FLY_STREAM_V1'
    def __call__(self):
        return self._handle['entry/instrument/detector/data'][:]


class BulkMerlinDebug(BulkXspress):
    # This is for data take in 'capture' mode, only used for debugging
    # once.
    HANDLER_NAME = 'MERLIN_FLY'
    def __call__(self):
        return self._handle['entry/instrument/detector/data'][1:]


class MerlinFileStoreHDF5(FileStoreBase):

    _spec = 'AD_HDF5'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs.update([('auto_increment', 'Yes'),
                                ('array_counter', 0),
                                ('auto_save', 'Yes'),
                                ('num_capture', 0),  # will be updated later
                                (self.file_template, '%s%s_%6.6d.h5'),
                                (self.file_write_mode, 'Stream'),
                                # (self.compression, 'zlib'),
                                (self.capture, 1)
                                ])
        self._point_counter = None

    def unstage(self):
        self._point_counter = None
        return super().unstage()

    def make_filename(self):
        filename = new_short_uid()
        formatter = datetime.datetime.now().strftime
        write_path = formatter(self.write_path_template)
        read_path = formatter(self.read_path_template)

        fn, read_path, write_path = filename, read_path, write_path
        return fn, read_path, write_path

    @property
    def filestore_spec(self):
        if self.parent._mode is SRXMode.fly:
            return BulkMerlin.HANDLER_NAME
        return 'AD_HDF5'

    def generate_datum(self, key, timestamp, datum_kwargs):
        if self.parent._mode is SRXMode.fly:
            return super().generate_datum(key, timestamp, datum_kwargs)
        elif self.parent._mode is SRXMode.step:
            i = next(self._point_counter)
            datum_kwargs = datum_kwargs or {}
            datum_kwargs.update({'point_number': i})
            return super().generate_datum(key, timestamp, datum_kwargs)

    def stage(self):
        # Make a filename.
        filename, read_path, write_path = self.make_filename()

        # Ensure we do not have an old file open.
        _TIMEOUT = 2
        self.capture.set(0, timeout=_TIMEOUT).wait()
        # These must be set before parent is staged (specifically
        # before capture mode is turned on. They will not be reset
        # on 'unstage' anyway.
        self.file_path.set(write_path, timeout=_TIMEOUT).wait()
        self.file_name.set(filename, timeout=_TIMEOUT).wait()
        self.file_number.set(0, timeout=_TIMEOUT).wait()
        staged = super().stage()

        # AD does this same templating in C, but we can't access it
        # so we do it redundantly here in Python.
        # file_number is *next* iteration
        self._fn = self.file_template.get() % (read_path,
                                               filename,
                                               self.file_number.get() - 1)
        self._fp = read_path
        if not self.file_path_exists.get():
            raise IOError("Path %s does not exist on IOC."
                          "" % self.file_path.get())

        if self.parent._mode is SRXMode.fly:
            res_kwargs = {}
        else:
            res_kwargs = {'frame_per_point': 1}
            self._point_counter = itertools.count()

        logger.debug("Inserting resource with filename %s", self._fn)
        self._generate_resource(res_kwargs)

        return staged


class HDF5PluginWithFileStoreMerlin(HDF5Plugin, MerlinFileStoreHDF5):

    def stage(self):
        if np.array(self.array_size.get()).sum() == 0:
            raise Exception("you must warmup the hdf plugin via the `warmup()` "
                            "method on the hdf5 plugin.")

        return super().stage()
    

# Adding operating energy to Merlin
class SRXMerlinDetectorCam(CamBase):
    acquire = ADComponent(EpicsSignal,'Acquire')
    quad_merlin_mode = ADComponent(EpicsSignalWithRBV,'QuadMerlinMode')
    operating_energy = ADComponent(EpicsSignal, 'OperatingEnergy')
    pass


class SRXMerlinDetector(AreaDetector):
    cam = Cpt(SRXMerlinDetectorCam, 'cam1:',
              read_attrs=[],
              configuration_attrs=['image_mode', 'trigger_mode',
                                   'acquire_time', 'acquire_period'],
              )
    

# class SRXMerlin(SingleTrigger, MerlinDetector):
class SRXMerlin(SingleTrigger, SRXMerlinDetector):
    total_points = Cpt(Signal,
                       value=1,
                       doc="The total number of points to be taken")
    fly_next = Cpt(Signal,
                   value=False,
                   doc="latch to put the detector in 'fly' mode")

    path_start = "/nsls2/data/srx/"

    def root_path_str():
        data_session = RE.md["data_session"]
        cycle = RE.md["cycle"]
        if "Commissioning" in get_proposal_type():
            root_path = f"proposals/commissioning/{data_session}/assets/merlin/"
        else:
            root_path = f"proposals/{cycle}/{data_session}/assets/merlin/"
        return root_path

    def path_template_str(root_path):
        path_template = "%Y/%m/%d/"
        return root_path + path_template

    hdf5 = Cpt(HDF5PluginWithFileStoreMerlin, 'HDF1:',
                read_attrs=[],
                configuration_attrs=[],
                write_path_template=path_start + path_template_str(root_path_str()),
                read_path_template=path_start + path_template_str(root_path_str()),
                root=path_start+root_path_str())

    stats1 = Cpt(StatsPlugin, 'Stats1:')
    stats2 = Cpt(StatsPlugin, 'Stats2:')
    stats3 = Cpt(StatsPlugin, 'Stats3:')
    stats4 = Cpt(StatsPlugin, 'Stats4:')
    stats5 = Cpt(StatsPlugin, 'Stats5:')
    proc1 = Cpt(ProcessPlugin, 'Proc1:')
    transform1 = Cpt(TransformPlugin, 'Trans1:')

    roi1 = Cpt(ROIPlugin, 'ROI1:')
    roi2 = Cpt(ROIPlugin, 'ROI2:')
    roi3 = Cpt(ROIPlugin, 'ROI3:')
    roi4 = Cpt(ROIPlugin, 'ROI4:')

    # def __init__(self, prefix, *, configuration_attrs=None, read_attrs=None,
    #              **kwargs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = SRXMode.step

    def stop(self, success=False):
        ret = super().stop(success=success)
        self.hdf5.stop()
        return ret

    def stage(self):
        # EJM: Clear counter for consistency with Xspress3
        _TIMEOUT = 2
        self.cam.array_counter.set(0, timeout=_TIMEOUT).wait()

        # do the latching
        if self.fly_next.get():
            self.fly_next.put(False)
            self.stage_sigs[self.cam.trigger_mode] = 2  # Trigger Start Rising
            self._mode = SRXMode.fly
        else:
            # Set trigger mode
            self.stage_sigs[self.cam.trigger_mode] = 0

            # Make sure we respect whatever the exposure time is set to
            count_time = self.cam.acquire_time.get()
            if count_time is not None:
                self.stage_sigs[self.cam.acquire_time] = count_time
                self.stage_sigs[self.cam.acquire_period] = count_time + 0.005
            self._mode = SRXMode.step

        return super().stage()

    def unstage(self):
        try:
            ret = super().unstage()
        finally:
            self._mode = SRXMode.step
        return ret


try:
    print("  Define merlin...", end="")
    merlin = SRXMerlin('XF:05IDD-ES{Merlin:1}',
                       name='merlin',
                       read_attrs=['hdf5', 'cam', 'stats1'])
    print("done")
    merlin.hdf5.read_attrs = []
    if np.array(merlin.cam.array_size.get()).sum() == 0:
        print("  Warmup...", end="", flush=True)
        merlin.hdf5.warmup()
        print("done")
except TimeoutError:
    print('\nCannot connect to Merlin. Continuing without device.\n')
    merlin = None
except Exception:
    print('\nUnexpected error connecting to Merlin.\n',
          sys.exc_info()[0],
          end='\n\n')
