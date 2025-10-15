print(f'Loading {__file__}...')


import os
import h5py
import datetime

# import ophyd
from hxntools.detectors.dexela import (DexelaDetector,)
from nslsii.detectors.xspress3 import (logger, )
from databroker.assets.handlers import HandlerBase
from ophyd.areadetector.filestore_mixins import (FileStoreIterativeWrite,
                                                 FileStoreHDF5IterativeWrite,
                                                 FileStoreTIFFSquashing,
                                                 FileStoreTIFF,
                                                 FileStoreHDF5,
                                                 new_short_uid,
                                                 FileStoreBase,
                                                 )
from ophyd.areadetector.trigger_mixins import SingleTrigger
from ophyd.areadetector import (AreaDetector, PixiradDetectorCam, ImagePlugin,
                                TIFFPlugin, StatsPlugin, HDF5Plugin,
                                ProcessPlugin, ROIPlugin, TransformPlugin,
                                OverlayPlugin)
from ophyd import Component as Cpt


class BulkDexela(HandlerBase):
    HANDLER_NAME = 'DEXELA_FLY_V1'

    def __init__(self, resource_fn):
        self._handle = h5py.File(resource_fn, 'r')

    def __call__(self):
        return self._handle['entry/instrument/detector/data'][:]

    def close(self):
        self._handle.close()
        self._handle = None
        super().close()


# db.reg.register_handler(BulkDexela.HANDLER_NAME, BulkDexela,
#                         overwrite=True)


class DexelaFileStoreHDF5(FileStoreBase):
    @property
    def filestore_spec(self):
        if self.parent._mode is SRXMode.fly:
            return BulkDexela.HANDLER_NAME
        # return 'TPX_HDF5'
        return 'AD_HDF5'

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

    def generate_datum(self, key, timestamp, datum_kwargs):
        if self.parent._mode is SRXMode.fly:
            return super().generate_datum(key, timestamp, datum_kwargs)
        elif self.parent._mode is SRXMode.step:
            i = next(self._point_counter)
            datum_kwargs = datum_kwargs or {}
            datum_kwargs.update({'point_number': i})
            return super().generate_datum(key, timestamp, datum_kwargs)

    def stage(self):
        # Timeout
        _TIMEOUT = 10
        # Make a filename.
        filename, read_path, write_path = self.make_filename()

        # Ensure we do not have an old file open.
        self.capture.set(0, timeout=_TIMEOUT).wait()

        # These must be set before parent is staged (specifically
        # before capture mode is turned on. They will not be reset
        # on 'unstage' anyway.
        self.file_path.set(write_path, timeout=_TIMEOUT).wait()
        self.file_name.put(filename)
        # AMK does not like this
        if self.file_number.get() != 0:
            self.file_number.set(0, timeout=_TIMEOUT).wait()

        if self.parent._mode is SRXMode.step:
            self.num_capture.set(self.parent.total_points.get(), timeout=_TIMEOUT).wait()

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
            self.parent.cam.num_images.set(1, timeout=_TIMEOUT).wait()
            res_kwargs = {'frame_per_point': 1}

            self._point_counter = itertools.count()

        logger.debug("Inserting resource with filename %s", self._fn)
        self._generate_resource(res_kwargs)

        return staged


class DexelaHDFWithFileStore(HDF5Plugin, DexelaFileStoreHDF5):
    def stage(self):
        if np.array(self.array_size.get()).sum() == 0:
            raise Exception("You must warmup the hdf plugin via the `warmup()`"
                            " method on the hdf5 plugin.")

        return super().stage()

    def warmup(self):
        """
        A convenience method for 'priming' the plugin.

        The plugin has to 'see' one acquisition before it is ready to capture.
        This sets the array size, etc.

        This needs to be redefined because there is no trigger_mode = Internal
        """
        self.enable.set(1).wait()
        sigs = OrderedDict(
            [
                (self.parent.cam.array_callbacks, 1),
                (self.parent.cam.image_mode, "Single"),
                (self.parent.cam.trigger_mode, "Int. Free Run"),
                # just in case the acquisition time is set very long...
                (self.parent.cam.acquire_time, 1),
                (self.parent.cam.acquire_period, 1),
                (self.parent.cam.acquire, 1),
            ]
        )

        original_vals = {sig: sig.get() for sig in sigs}

        for sig, val in sigs.items():
            ttime.sleep(0.1)  # abundance of caution
            # print(f"{sig=}\t{val=}")
            sig.set(val).wait()

        ttime.sleep(2)  # wait for acquisition

        for sig, val in reversed(list(original_vals.items())):
            ttime.sleep(0.1)
            sig.set(val).wait()


class SRXDexelaDetector(SingleTrigger, DexelaDetector):
    total_points = Cpt(Signal,
                       value=1,
                       doc="The total number of points to be taken")
    path_write_start = "W:\\"
    path_read_start = "/nsls2/data/srx/"

    def root_path_str():
        data_session = RE.md["data_session"]
        cycle = RE.md["cycle"]
        if "Commissioning" in get_proposal_type():
            root_path = f"proposals/commissioning/{data_session}/assets/dexela/"
        else:
            root_path = f"proposals/{cycle}/{data_session}/assets/dexela/"
        return root_path

    def path_template_str(root_path):
        path_template = "%Y/%m/%d/"
        return root_path + path_template

    hdf5 = Cpt(DexelaHDFWithFileStore, 'HDF1:',
               read_attrs=[],
               configuration_attrs=[],
               write_path_template=path_write_start + path_template_str(root_path_str()).replace("/", "\\"),
               read_path_template=path_read_start + path_template_str(root_path_str()),
               root=path_read_start+root_path_str())
    # this is used as a latch to put the xspress3 into 'bulk' mode
    # for fly scanning.  Do this is a signal (rather than as a local variable
    # or as a method so we can modify this as part of a plan
    fly_next = Cpt(Signal, value=False)

    proc1 = Cpt(ProcessPlugin, 'Proc1:')
    roi1 = Cpt(ROIPlugin, 'ROI1:')
    roi2 = Cpt(ROIPlugin, 'ROI2:')
    stats1 = Cpt(StatsPlugin, 'Stats1:', read_attrs=['total'])
    stats2 = Cpt(StatsPlugin, 'Stats2:', read_attrs=['total'])
    # stats1.read_attrs(['total'])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = SRXMode.step
        # self.cam.trigger_mode = EpicsSignalWithRBV("XF:05IDD-ES{Dexela:1}cam1:TriggerMode")

    def stage(self):
        # EJM: Clear counter for consistency with Xspress3
        _TIMEOUT = 2
        self.cam.array_counter.set(0, timeout=_TIMEOUT).wait()

        # do the latching
        if self.fly_next.get():
            self.fly_next.put(False)
            self._mode = SRXMode.fly

        self.cam.stage_sigs['image_mode'] = 'Multiple'
        if self._mode is SRXMode.fly:
            # self.cam.stage_sigs['trigger_mode'] = 'Ext. Edge Single'
            self.cam.stage_sigs['trigger_mode'] = 'Ext. Bulb'
        else:
            self.cam.stage_sigs['trigger_mode'] = 'Int. Fixed Rate'

        return super().stage()

    def unstage(self):
        try:
            ret = super().unstage()
        finally:
            self._mode = SRXMode.step
        return ret


try:
    dexela = SRXDexelaDetector('XF:05IDD-ES{Dexela:1}', name='dexela')
    dexela.read_attrs = ['hdf5']
    # dexela.read_attrs = ['hdf5', 'stats1', 'stats2']
    # Automatically warmup if necessary
    if np.array(dexela.cam.array_size.get()).sum() == 0:
        print("  Warmup...", end="", flush=True)
        dexela.hdf5.warmup()
        print("done")
except TimeoutError:
    dexela = None
    print('\nCannot connect to Dexela. Continuing without device.\n')
except Exception:
    dexela = None
    print('\nUnexpected error connecting to Dexela.\n',
          sys.exc_info()[0],
          end='\n\n')
