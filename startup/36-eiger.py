print(f'Loading {__file__}...')

import time as ttime
from ophyd import CamBase, AreaDetector, ProcessPlugin, TransformPlugin, ROIPlugin
from ophyd.areadetector.base import ADComponent, EpicsSignal, EpicsSignalWithRBV
from hxntools.detectors.trigger_mixins import HxnModalTrigger
from hxntools.detectors.merlin import HDF5PluginWithFileStore
from collections import OrderedDict


# # class SRXEigerModalTrigger(HxnModalTrigger):

# #     def mode_internal(self):
# #         super().mode_internal()

# #         cam = self.cam
# #         cam.stage_sigs[cam.num_images] = 1
# #         cam.stage_sigs[cam.image_mode] = 'Single'
# #         cam.stage_sigs[cam.trigger_mode] = 'Internal Series'

# #     def mode_external(self):
# #         super().mode_external()
# #         total_points = self.mode_settings.total_points.get()

# #         cam = self.cam
# #         cam.stage_sigs[cam.num_images] = 1
# #         cam.stage_sigs[cam.num_triggers] = total_points
# #         cam.stage_sigs[cam.image_mode] = 0  # 'Multiple'
# #         cam.stage_sigs[cam.trigger_mode] = 3  # 'External Enable'

# class EigerDetectorCam(CamBase):
#     num_triggers = ADComponent(EpicsSignalWithRBV, 'NumTriggers')
#     beam_center_x = ADComponent(EpicsSignalWithRBV, 'BeamX')
#     beam_center_y = ADComponent(EpicsSignalWithRBV, 'BeamY')
#     wavelength = ADComponent(EpicsSignalWithRBV, 'Wavelength')
#     det_distance = ADComponent(EpicsSignalWithRBV, 'DetDist')
#     threshold_energy = ADComponent(EpicsSignalWithRBV, 'ThresholdEnergy')
#     photon_energy = ADComponent(EpicsSignalWithRBV, 'PhotonEnergy')
#     manual_trigger = ADComponent(EpicsSignalWithRBV, 'ManualTrigger')  # the checkbox
#     special_trigger_button = ADComponent(EpicsSignal, 'Trigger')  # the button next to 'Start' and 'Stop'
#     auto_summation = ADComponent(EpicsSignal, 'AutoSummation')  # the button next to 'Start' and 'Stop'

#     def ensure_nonblocking(self):
#         for c in self.parent.component_names:
#             cpt = getattr(self.parent, c)
#             if cpt is self:
#                 continue
#             if hasattr(cpt, 'ensure_nonblocking'):
#                 cpt.ensure_nonblocking()


# class EigerDetector(AreaDetector):
#     cam = Cpt(EigerDetectorCam, 'cam1:',
#               read_attrs=[],
#               configuration_attrs=['image_mode', 'trigger_mode',
#                                    'acquire_time', 'acquire_period'],
#               )


# class HDF5PluginWithFileStoreEiger(HDF5PluginWithFileStore):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.stage_sigs.update([(self.compression, 'szip'),
#                                 (self.queue_size, 10000)])


#         # 'swmr_mode' must be set first. Rearrange 'stage_sigs'.
#         # self.stage_sigs[self.swmr_mode] = 1
#         # self.stage_sigs[self.num_frames_flush] = 1  # Set later
#         # self.stage_sigs.move_to_end(self.num_frames_flush, last=False)
#         # self.stage_sigs.move_to_end(self.swmr_mode, last=False)

#     def stage(self):
#         if np.array(self.array_size.get()).sum() == 0:
#             raise Exception("you must warmup the hdf plugin via the `warmup()` "
#                             "method on the hdf5 plugin.")

#         # if self.frame_per_point:
#         #     self.stage_sigs[self.num_frames_flush] = self.frame_per_point

#         return super().stage()

#     def describe(self):
#         desc = super().describe()

#         # Replace the shape for 'eiger2_image'. Height and width should be acquired directly
#         # from HDF5 plugin, since the size of the image could be restricted by ROI.
#         # Number of images is returned as 1, so replace it with the number of triggers (for flyscan).
#         for k, v in desc.items():
#             if k.endswith("_image") and ("shape" in v):
#                 height = self.height.get()
#                 width = self.width.get()
#                 # Generated shape is valid for flyscan using 'External Enable' triggering mode
#                 num_triggers = self.parent.cam.num_triggers.get()
#                 orig_shape = v["shape"]
#                 v["shape"] = (num_triggers, height, width)
#                 print(f"Descriptor: shape of {k!r} was updated. The shape {orig_shape} was replaced by {v['shape']}")

#         return desc

#     def warmup(self, acquire_time=1):
#         """
#         A convenience method for 'priming' the plugin.

#         The plugin has to 'see' one acquisition before it is ready to capture.
#         This sets the array size, etc.

#         Parameters
#         ----------
#         acquire_time: float
#             Exposure time for warmup, s
#         """
#         self.enable.set(1).wait()
#         sigs = OrderedDict(
#             [
#                 # (self.file_write_mode, "Capture"),
#                 # (self.file_write_mode, "Single"),
#                 # (self.parent.roi1.enable, 1),
#                 (self.parent.cam.array_callbacks, 0),
#                 (self.parent.cam.image_mode, "Single"),
#                 (self.parent.cam.trigger_mode, "Internal Series"),
#                 (self.parent.cam.manual_trigger, "Disable"),
#                 (self.parent.cam.num_triggers, 1),
#                 (self.parent.cam.acquire_period, acquire_time),  # Adjusted once acquire_time is set
#                 (self.parent.cam.acquire_time, acquire_time),
#                 (self.parent.cam.acquire, 1),
#             ]
#         )

#         original_vals = {sig: sig.get() for sig in sigs}

#         for sig, val in sigs.items():
#             ttime.sleep(0.1)  # abundance of caution
#             sig.set(val).wait()

#         ttime.sleep(acquire_time + 1)  # wait for acquisition

#         for sig, val in reversed(list(original_vals.items())):
#             ttime.sleep(0.1)
#             sig.set(val).wait()


# class SRXEigerDetector(HxnModalTrigger, EigerDetector):
#     proc1 = Cpt(ProcessPlugin, 'Proc1:')
#     stats1 = Cpt(StatsPluginV33, 'Stats1:')
#     stats2 = Cpt(StatsPluginV33, 'Stats2:')
#     stats3 = Cpt(StatsPluginV33, 'Stats3:')
#     stats4 = Cpt(StatsPluginV33, 'Stats4:')
#     stats5 = Cpt(StatsPluginV33, 'Stats5:')
#     transform1 = Cpt(TransformPlugin, 'Trans1:')
#     roi1 = Cpt(ROIPlugin, 'ROI1:')
#     roi2 = Cpt(ROIPlugin, 'ROI2:')
#     roi3 = Cpt(ROIPlugin, 'ROI3:')
#     roi4 = Cpt(ROIPlugin, 'ROI4:')

#     hdf5 = Cpt(HDF5PluginWithFileStoreEiger, 'HDF1:',
#                read_attrs=[],
#                configuration_attrs=[],
#                write_path_template='/data/%Y/%m/%d/',
#                root='/data',
#                reg=db.reg)

#     def __init__(self, prefix, *, read_attrs=None, configuration_attrs=None,
#                  **kwargs):
#         if read_attrs is None:
#             read_attrs = ['hdf5', 'cam']
#         if configuration_attrs is None:
#             configuration_attrs = ['hdf5', 'cam']

#         if 'hdf5' not in read_attrs:
#             # ensure that hdf5 is still added, or data acquisition will fail
#             read_attrs = list(read_attrs) + ['hdf5']

#         super().__init__(prefix, configuration_attrs=configuration_attrs,
#                          read_attrs=read_attrs, **kwargs)

#         self.cam.ensure_nonblocking()

#     def mode_internal(self):
#         super().mode_internal()

#         cam = self.cam
#         cam.stage_sigs[cam.num_images] = 1
#         cam.stage_sigs[cam.num_triggers] = 1
#         cam.stage_sigs[cam.image_mode] = 'Single'
#         cam.stage_sigs[cam.trigger_mode] = 'Internal Series'

#         count_time = self.count_time.get()
#         if count_time is not None:
#             self.stage_sigs[self.cam.acquire_time] = count_time
#             self.stage_sigs[self.cam.acquire_period] = count_time + 0.005

#     def mode_external(self):
#         super().mode_external()

#         total_points = self.mode_settings.total_points.get()

#         cam = self.cam
#         cam.stage_sigs[cam.num_images] = 1
#         cam.stage_sigs[cam.num_triggers] = total_points
#         cam.stage_sigs[cam.image_mode] = 'Multiple'
#         cam.stage_sigs[cam.trigger_mode] = 'External Enable'

#         # When Eiger is in 'external enable' mode, the exposure time is used to
#         # set the bit depth of the detector. It is recommended that the exposure
#         # is set to the minimum expected exposure, so it should be set using plan
#         # parameters.
#         expected_exposure = 0.03
#         self.stage_sigs[self.cam.acquire_time] = expected_exposure
#         self.stage_sigs[self.cam.acquire_period] = expected_exposure

#         # self.cam.stage_sigs[self.cam.trigger_mode] = 'Trigger Enable'

# # class HxnEigerDetector(_HMD):
# #     stats1 = Cpt(StatsPluginV33, 'Stats1:')
# #     stats2 = Cpt(StatsPluginV33, 'Stats2:')
# #     stats3 = Cpt(StatsPluginV33, 'Stats3:')
# #     stats4 = Cpt(StatsPluginV33, 'Stats4:')
# #     stats5 = Cpt(StatsPluginV33, 'Stats5:')

# #     hdf5 = Cpt(_mhdf, 'HDF1:',
# #                read_attrs=[],
# #                configuration_attrs=[],
# #                write_path_template='/data/%Y/%m/%d/',
# #                root='/data',
# #                reg=db.reg)

# #     def mode_internal(self):
# #         super().mode_internal()

# #         cam = self.cam
# #         cam.stage_sigs[cam.num_images] = 1
# #         cam.stage_sigs[cam.image_mode] = 'Single'
# #         cam.stage_sigs[cam.trigger_mode] = 'Internal Series'

# #     def mode_external(self):
# #         super().mode_external()
# #         total_points = self.mode_settings.total_points.get()

# #         cam = self.cam
# #         cam.stage_sigs[cam.num_images] = 1
# #         cam.stage_sigs[cam.num_triggers] = total_points
# #         cam.stage_sigs[cam.image_mode] = 0  # 'Multiple'
# #         cam.stage_sigs[cam.trigger_mode] = 3  # 'External Enable'



### Rewritten for SRX

class EigerDetectorCam(CamBase):
    num_triggers = ADComponent(EpicsSignalWithRBV, 'NumTriggers')
    beam_center_x = ADComponent(EpicsSignalWithRBV, 'BeamX')
    beam_center_y = ADComponent(EpicsSignalWithRBV, 'BeamY')
    wavelength = ADComponent(EpicsSignalWithRBV, 'Wavelength')
    det_distance = ADComponent(EpicsSignalWithRBV, 'DetDist')
    threshold_energy = ADComponent(EpicsSignalWithRBV, 'ThresholdEnergy')
    photon_energy = ADComponent(EpicsSignalWithRBV, 'PhotonEnergy')
    manual_trigger = ADComponent(EpicsSignalWithRBV, 'ManualTrigger')  # the checkbox
    special_trigger_button = ADComponent(EpicsSignal, 'Trigger')  # the button next to 'Start' and 'Stop'
    auto_summation = ADComponent(EpicsSignal, 'AutoSummation')  # the button next to 'Start' and 'Stop'

    def ensure_nonblocking(self):
        for c in self.parent.component_names:
            cpt = getattr(self.parent, c)
            if cpt is self:
                continue
            if hasattr(cpt, 'ensure_nonblocking'):
                cpt.ensure_nonblocking()


class EigerDetector(AreaDetector):
    cam = Cpt(EigerDetectorCam, 'cam1:',
              read_attrs=[],
              configuration_attrs=['image_mode', 'trigger_mode',
                                   'acquire_time', 'acquire_period'],
              )


class EigerFileStoreHDF5(FileStoreBase):
    @property
    def filestore_spec(self):
        # if self.parent._mode is SRXMode.fly:
        #     return BulkDexela.HANDLER_NAME # nothing special yet
        return 'AD_HDF5'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs.update([('auto_increment', 'Yes'),
                                ('array_counter', 0),
                                ('auto_save', 'Yes'),
                                ('num_capture', 0),  # will be updated later
                                ('file_template', '%s%s_%6.6d.h5'),
                                ('file_write_mode', 'Stream'),
                                ('capture', 1)
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
            datum_kwargs = datum_kwargs or {}
            datum_kwargs.update({'point_number': 1})
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
            res_kwargs = {'frame_per_point': 1}
            # res_kwargs = {'frame_per_point' : self.parent.cam.num_triggers.get()}
        else:
            self.parent.cam.num_images.set(1, timeout=_TIMEOUT).wait()
            res_kwargs = {'frame_per_point': 1}

            self._point_counter = itertools.count()

        logger.debug("Inserting resource with filename %s", self._fn)
        self._generate_resource(res_kwargs)

        return staged


class EigerHDFWithFileStore(HDF5Plugin, EigerFileStoreHDF5):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stage_sigs.update([('compression', 'szip'),
                                ('queue_size', 10000)])

    def stage(self):
        if np.array(self.array_size.get()).sum() == 0:
            raise Exception("you must warmup the hdf plugin via the `warmup()` "
                            "method on the hdf5 plugin.")

        # if self.frame_per_point:
        #     self.stage_sigs[self.num_frames_flush] = self.frame_per_point

        return super().stage()
    
    def describe(self):
        desc = super().describe()

        # Replace the shape for 'eiger1_image'. Height and width should be acquired directly
        # from HDF5 plugin, since the size of the image could be restricted by ROI.
        # Number of images is returned as 1, so replace it with the number of triggers (for flyscan).
        for k, v in desc.items():
            if k.endswith("_image") and ("shape" in v):
                height = self.height.get()
                width = self.width.get()
                # Generated shape is valid for flyscan using 'External Enable' triggering mode
                num_triggers = self.parent.cam.num_triggers.get()
                orig_shape = v["shape"]
                v["shape"] = (num_triggers, height, width)
                print(f"Descriptor: shape of {k!r} was updated. The shape {orig_shape} was replaced by {v['shape']}")

        return desc

    def warmup(self, acquire_time=1):
        """
        A convenience method for 'priming' the plugin.

        The plugin has to 'see' one acquisition before it is ready to capture.
        This sets the array size, etc.

        Parameters
        ----------
        acquire_time: float
            Exposure time for warmup, s
        """
        self.enable.set(1).wait()
        sigs = OrderedDict(
            [
                (self.parent.cam.array_callbacks, 0),
                (self.parent.cam.image_mode, "Single"),
                (self.parent.cam.trigger_mode, "Internal Series"),
                (self.parent.cam.manual_trigger, "Disable"),
                (self.parent.cam.num_triggers, 1),
                (self.parent.cam.acquire_period, acquire_time),  # Adjusted once acquire_time is set
                (self.parent.cam.acquire_time, acquire_time),
                (self.parent.cam.acquire, 1),
            ]
        )

        original_vals = {sig: sig.get() for sig in sigs}

        for sig, val in sigs.items():
            ttime.sleep(0.1)  # abundance of caution
            sig.set(val).wait()

        ttime.sleep(acquire_time + 1)  # wait for acquisition

        for sig, val in reversed(list(original_vals.items())):
            ttime.sleep(0.1)
            sig.set(val).wait()


class SRXEigerDetector(SingleTrigger, EigerDetector):
    proc1 = Cpt(ProcessPlugin, 'Proc1:')
    stats1 = Cpt(StatsPluginV33, 'Stats1:')
    stats2 = Cpt(StatsPluginV33, 'Stats2:')
    stats3 = Cpt(StatsPluginV33, 'Stats3:')
    stats4 = Cpt(StatsPluginV33, 'Stats4:')
    stats5 = Cpt(StatsPluginV33, 'Stats5:')
    transform1 = Cpt(TransformPlugin, 'Trans1:')
    roi1 = Cpt(ROIPlugin, 'ROI1:')
    roi2 = Cpt(ROIPlugin, 'ROI2:')
    roi3 = Cpt(ROIPlugin, 'ROI3:')
    roi4 = Cpt(ROIPlugin, 'ROI4:')

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
            root_path = f"proposals/commissioning/{data_session}/assets/eiger1/"
        else:
            root_path = f"proposals/{cycle}/{data_session}/assets/eiger1/"
        return root_path

    def path_template_str(root_path):
        path_template = "%Y/%m/%d/"
        return root_path + path_template

    hdf5 = Cpt(EigerHDFWithFileStore, 'HDF1:',
                read_attrs=[],
                configuration_attrs=[],
                # write_path_template=path_start + path_template_str(root_path_str()),
                # read_path_template=path_start + path_template_str(root_path_str()),
                # root=path_start+root_path_str()
                write_path_template=path_start + f'proposals/commissioning/pass-318777/eiger1/',
                read_path_template=path_start + f'proposals/commissioning/pass-318777/eiger1/',
                root=path_start + f'proposals/commissioning/pass-318777/eiger1/'
                )

    def __init__(self, prefix, *, read_attrs=None, configuration_attrs=None,
                 **kwargs):
        if read_attrs is None:
            read_attrs = ['hdf5', 'cam']
        if configuration_attrs is None:
            configuration_attrs = ['hdf5', 'cam']

        if 'hdf5' not in read_attrs:
            # ensure that hdf5 is still added, or data acquisition will fail
            read_attrs = list(read_attrs) + ['hdf5']

        super().__init__(prefix, configuration_attrs=configuration_attrs,
                         read_attrs=read_attrs, **kwargs)
        self._mode = SRXMode.step
        self.cam.ensure_nonblocking()


    def stage(self):
        # EJM: Clear counter for consistency with Xspress3
        _TIMEOUT = 2
        self.cam.array_counter.set(0, timeout=_TIMEOUT).wait()
        total_points = self.total_points.get()

        # do the latching
        if self.fly_next.get():
            self.fly_next.put(False)
            self._mode = SRXMode.fly

        if self._mode is SRXMode.fly:
            self.cam.stage_sigs['num_images'] = 1
            self.cam.stage_sigs['num_triggers'] = total_points
            self.cam.stage_sigs['image_mode'] = 'Multiple'
            self.cam.stage_sigs['trigger_mode'] = 'External Enable'
        else:
            self.cam.stage_sigs['num_images'] = 1
            self.cam.stage_sigs['num_triggers'] = 1
            self.cam.stage_sigs['image_mode'] = 'Single'
            self.cam.stage_sigs['trigger_mode'] = 'Internal Series'

        return super().stage()


    def unstage(self):
        try:
            ret = super().unstage()
        finally:
            self._mode = SRXMode.step
        return ret


try:
    eiger1 = SRXEigerDetector('XF:05IDD-ES{Det:Eig1M}', name='eiger1',
                              #image_name='eiger1',
                              read_attrs=['hdf5',
                                          # 'cam',
                                          # 'stats1'
                                          ])
    eiger1.hdf5.read_attrs = []
    eiger1.cam.auto_summation.set('Enable')
    if np.array(eiger1.cam.array_size.get()).sum() == 0:
        print("  Warmup...", end="", flush=True)
        eiger1.hdf5.warmup()
        print("done")
except TimeoutError:
    print(f'\nCannot connect to eiger1. Continuing without device.\n')
except Exception as ex:
    print(ex, end='\n\n')
