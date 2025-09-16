print(f"Loading {__file__!r} ...")

import time as ttime
from ophyd import CamBase, AreaDetector, ProcessPlugin, TransformPlugin, ROIPlugin
from ophyd.areadetector.base import ADComponent, EpicsSignal, EpicsSignalWithRBV
from hxntools.detectors.trigger_mixins import HxnModalTrigger
from hxntools.detectors.merlin import HDF5PluginWithFileStore
from collections import OrderedDict


# class SRXEigerModalTrigger(HxnModalTrigger):

#     def mode_internal(self):
#         super().mode_internal()

#         cam = self.cam
#         cam.stage_sigs[cam.num_images] = 1
#         cam.stage_sigs[cam.image_mode] = 'Single'
#         cam.stage_sigs[cam.trigger_mode] = 'Internal Series'

#     def mode_external(self):
#         super().mode_external()
#         total_points = self.mode_settings.total_points.get()

#         cam = self.cam
#         cam.stage_sigs[cam.num_images] = 1
#         cam.stage_sigs[cam.num_triggers] = total_points
#         cam.stage_sigs[cam.image_mode] = 0  # 'Multiple'
#         cam.stage_sigs[cam.trigger_mode] = 3  # 'External Enable'

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

class HDF5PluginWithFileStoreEiger(HDF5PluginWithFileStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stage_sigs.update([(self.compression, 'szip'),
                                (self.queue_size, 10000)])


        # 'swmr_mode' must be set first. Rearrange 'stage_sigs'.
        # self.stage_sigs[self.swmr_mode] = 1
        # self.stage_sigs[self.num_frames_flush] = 1  # Set later
        # self.stage_sigs.move_to_end(self.num_frames_flush, last=False)
        # self.stage_sigs.move_to_end(self.swmr_mode, last=False)

    def stage(self):
        if np.array(self.array_size.get()).sum() == 0:
            raise Exception("you must warmup the hdf plugin via the `warmup()` "
                            "method on the hdf5 plugin.")

        # if self.frame_per_point:
        #     self.stage_sigs[self.num_frames_flush] = self.frame_per_point

        return super().stage()

    def describe(self):
        desc = super().describe()

        # Replace the shape for 'eiger2_image'. Height and width should be acquired directly
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
                # (self.file_write_mode, "Capture"),
                # (self.file_write_mode, "Single"),
                (self.parent.roi1.enable, 1),
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


class SRXEigerDetector(HxnModalTrigger, EigerDetector):
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

    hdf5 = Cpt(HDF5PluginWithFileStoreEiger, 'HDF1:',
               read_attrs=[],
               configuration_attrs=[],
               write_path_template='/data/%Y/%m/%d/',
               root='/data',
               reg=db.reg)

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

        self.cam.ensure_nonblocking()

    def mode_internal(self):
        super().mode_internal()

        cam = self.cam
        cam.stage_sigs[cam.num_images] = 1
        cam.stage_sigs[cam.num_triggers] = 1
        cam.stage_sigs[cam.image_mode] = 'Single'
        cam.stage_sigs[cam.trigger_mode] = 'Internal Series'

        count_time = self.count_time.get()
        if count_time is not None:
            self.stage_sigs[self.cam.acquire_time] = count_time
            self.stage_sigs[self.cam.acquire_period] = count_time + 0.005

    def mode_external(self):
        super().mode_external()

        total_points = self.mode_settings.total_points.get()

        cam = self.cam
        cam.stage_sigs[cam.num_images] = 1
        cam.stage_sigs[cam.num_triggers] = total_points
        cam.stage_sigs[cam.image_mode] = 'Multiple'
        cam.stage_sigs[cam.trigger_mode] = 'External Enable'

        # When Eiger is in 'external enable' mode, the exposure time is used to
        # set the bit depth of the detector. It is recommended that the exposure
        # is set to the minimum expected exposure, so it should be set using plan
        # parameters.
        expected_exposure = 0.03
        self.stage_sigs[self.cam.acquire_time] = expected_exposure
        self.stage_sigs[self.cam.acquire_period] = expected_exposure

        # self.cam.stage_sigs[self.cam.trigger_mode] = 'Trigger Enable'

# class HxnEigerDetector(_HMD):
#     stats1 = Cpt(StatsPluginV33, 'Stats1:')
#     stats2 = Cpt(StatsPluginV33, 'Stats2:')
#     stats3 = Cpt(StatsPluginV33, 'Stats3:')
#     stats4 = Cpt(StatsPluginV33, 'Stats4:')
#     stats5 = Cpt(StatsPluginV33, 'Stats5:')

#     hdf5 = Cpt(_mhdf, 'HDF1:',
#                read_attrs=[],
#                configuration_attrs=[],
#                write_path_template='/data/%Y/%m/%d/',
#                root='/data',
#                reg=db.reg)

#     def mode_internal(self):
#         super().mode_internal()

#         cam = self.cam
#         cam.stage_sigs[cam.num_images] = 1
#         cam.stage_sigs[cam.image_mode] = 'Single'
#         cam.stage_sigs[cam.trigger_mode] = 'Internal Series'

#     def mode_external(self):
#         super().mode_external()
#         total_points = self.mode_settings.total_points.get()

#         cam = self.cam
#         cam.stage_sigs[cam.num_images] = 1
#         cam.stage_sigs[cam.num_triggers] = total_points
#         cam.stage_sigs[cam.image_mode] = 0  # 'Multiple'
#         cam.stage_sigs[cam.trigger_mode] = 3  # 'External Enable'



try:
    eiger1 = SRXEigerDetector('XF:05IDD-ES{Det:Eig1M}', name='eiger1',
                              image_name='eiger1',
                              read_attrs=['hdf5', 'cam', 'stats1'])
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
