print(f"Loading {__file__}...")

import os
import h5py
import sys
import numpy as np
import time as ttime
import itertools
from uuid import uuid4
from event_model import compose_resource

from ophyd.areadetector.plugins import PluginBase
from ophyd import Signal, EpicsSignal, DeviceStatus
from ophyd import Component as Cpt
from ophyd.areadetector.filestore_mixins import FileStorePluginBase
from ophyd.device import Staged
from ophyd.sim import NullStatus
from enum import Enum
from collections import deque, OrderedDict

from ophyd.areadetector import Xspress3Detector
from nslsii.areadetector.xspress3 import (
    build_xspress3_class,
    Xspress3HDF5Plugin,
    Xspress3Trigger,
    Xspress3FileStore,
)

# this is the community IOC package
from nslsii.areadetector.xspress3 import (
    build_xspress3_class
)

# JL
# set up some logging to help with development
#import logging
#console_log_handler = logging.StreamHandler(stream=sys.stdout)
#console_log_handler.setLevel("DEBUG")
#console_log_handler.setLevel(logging.DEBUG)
#console_log_handler.setFormatter(
#    logging.Formatter("[%(levelname)s %(asctime)s.%(msecs)03d %(module)15s:%(lineno)5d] %(message)s")
#)
#logging.getLogger("bluesky").addHandler(console_log_handler)
#logging.getLogger("bluesky").setLevel(logging.DEBUG)


try:
    from area_detector_handlers import HandlerBase
    from area_detector_handlers.handlers import Xspress3HDF5Handler
except ImportError:
    from databroker.assets.handlers import Xspress3HDF5Handler, HandlerBase


def get_me_the_cam(obj):
    # Helper function for interopability between QD and Community IOCs
    return obj.cam if hasattr(obj, 'cam') else obj.settings

class SRXMode(Enum):
    step = 1
    fly = 2


class BulkXspress(HandlerBase):
    HANDLER_NAME = "XSP3_FLY"

    def __init__(self, resource_fn):
        self._handle = h5py.File(resource_fn, "r")

    def __call__(self):
        return self._handle["entry/instrument/detector/data"][:]


# db.reg.register_handler(BulkXspress.HANDLER_NAME, BulkXspress, overwrite=True)


# class CommunityXspress3FileStoreFlyable(Xspress3FileStore):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     @property
#     def filestore_res(self):
#         raise Exception("don't wanCommunitySRXXspressTriggert to be here")
#         return self._filestore_res

#     @property
#     def filestore_spec(self):
#         if self.parent._mode is SRXMode.fly:
#             return BulkXspress.HANDLER_NAME
#         return Xspress3HDF5Handler.HANDLER_NAME

#     def generate_datum(self, key, CommunitySRXXspressTriggertimestamp, datum_kwargs):
#         if self.parent._mode is SRXMode.step:
#             print("Datum: STEP mode")
#             return super().generate_datum(key, timestamp, datum_kwargs)
#         elif self.parent._mode is SRXMode.fly:
#             print("Datum: FLY mode")
#             # we are doing something _very_ dirty here to skip a level
#             # of the inheritance
#             # this is brittle is if the MRO changes we may not hit all
#             # the level we expect to
#             return FileStorePluginBase.generate_datum(
#                 self, key, timestamp, datum_kwargs
#             )

#     def warmup(self):
#         """
#         A convenience method for 'priming' the plugin.
#         The plugin has to 'see' one acquisition before it is ready to capture.
#         This sets the array size, etc.

#         NOTE : this comes from:
#             https://github.com/NSLS-II/ophyd/blob/master/ophyd/areadetector/plugins.py
#         We had to replace "cam" with "settings" here.
#         Also modified the stage sigs.

#         """
#         print("  Warming up the hdf5 plugin...", end="", flush=True)
#         # set_and_wait(self.enable, 1)  // deprecated
#         self.enable.set(1).wait()
#         sigs = OrderedDict(
#             [
#                 (self.parent.cam.array_callbacks, 1),
#                 (self.parent.cam.image_mode, "Single"),
#                 (self.parent.cam.trigger_mode, "Internal"),
#                 # In case the acquisition time is set very long
#                 (self.parent.cam.acquire_time, 1),
#                 # (self.parent.cam.acquire_period, 1),
#                 (self.parent.cam.acquire, 1),
#             ]
#         )

#         original_vals = {sig: sig.get() for sig in sigs}

#         for sig, val in sigs.items():
#             ttime.sleep(0.1)  # abundance of caution
#             # set_and_wait(sig, val)  // deprecated
#             sig.set(val).wait()

#         ttime.sleep(2)  # wait for acquisition

#         for sig, val in reversed(list(original_vals.items())):
#             ttime.sleep(0.1)
#             # set_and_wait(sig, val)  // deprecated
#             sig.set(val).wait()
#         print("done")

#     def describe(self):
#         desc = super().describe()

#         if self.parent._mode is SRXMode.fly:
#             spec = {
#                 "external": "FileStore:",
#                 "dtype": "array",
#                 # TODO do not hard code
#                 "shape": (self.parent.cam.num_images.get(), 4, 4096),
#                 "source": self.prefix,
#             }
#             return {self.parent._f_key: spec}
#         else:
#             return super().describe()


class CommunitySRXXspressTrigger(Xspress3Trigger):
    # def trigger(self):
    #     print(f"  triggering xs3...")
    #     if self._staged != Staged.yes:
    #         raise RuntimeError("not staged")

    #     self._status = DeviceStatus(self)
    #     # the next line cauCommunitySRXXspressTriggerses a ~3s delay in the community IOC
    #     #self.cam.erase.put(1)
    #     print(f"  put acquire...")
    #     self.cam.acquire.put(1, wait=False)
    #     trigger_time = ttime.time()
    #     if self._mode is SRXMode.step:
    #         # community IOC ophyd xspress3
    #         print(f"  generate datum...")
    #         self.generate_datum(None, trigger_time, {})
    #         # quantum IOC ophyd xspress3
    #         #for sn in self.read_attrs:
    #         #    if sn.startswith("channel") and "." not in sn:
    #         #        ch = getattr(self, sn)
    #         #        self.dCommunitySRXXspressTriggerispatch(ch.name, trigger_time)
    #     elif self._mode is SRXMode.fly:
    #         self.generate_datum(self._f_key, trigger_time)
    #     else:
    #         raise Exception(f"unexpected mode {self._mode}")
    #     print(f"  increment trigger count...")
    #     self._abs_trigger_count += 1
    #     print(f"  return")
    #     return self._status

    # @property
    # def filestore_spec(self):
    #     fss = BulkXspress.HANDLER_NAME if self.parent._mode is SRXMode.fly else Xspress3HDF5Handler.HANDLER_NAME
    #     print(f"filestore_spec: {fss!r}")
    #     return fss


    def trigger(self):
        logger.debug("trigger")
        #print(f"  trigger xs3...")
        if self._staged != Staged.yes:
            raise RuntimeError(
                "tried to trigger Xspress3 with prefix {self.prefix} but it is not staged"
            )

        #print(f"  new acquire status...")
        self._acquire_status = self.new_acquire_status()
        #print(f"  put acquire")
        self.cam.acquire.put(1, wait=False)
        t0 = ttime.monotonic()
        while ("Acquiring Data" not in self.cam.status_message.get()):
            ttime.sleep(0.010)
            if (ttime.monotonic() - t0 > 10):
                raise TimeoutError
        trigger_time = ttime.time()

        # call generate_datum on all plugins
        self.generate_datum(
            key=None,
            timestamp=trigger_time,
            datum_kwargs={"frame": self._abs_trigger_count,
                          "channel": 1},
        )
        self._abs_trigger_count += 1

        #print(f"  return...")
        return self._acquire_status


class SrxXSP3Handler:
    XRF_DATA_KEY = "entry/instrument/detector/data"

    def __init__(self, filepath, **kwargs):
        self._filepath = filepath

    def __call__(self, **kwargs):
        with h5py.File(self._filepath, "r") as f:
            return np.asarray(f[self.XRF_DATA_KEY])


class Xspress3HDF5PluginWithRedis(Xspress3HDF5Plugin):
    "Subclass to determine file location based on proposal info in Redis"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._redis_dict = RunEngineRedisDict("info.srx.nsls2.bnl.gov")
        if kwargs["root_path"] is None:
            self.root_path.put(self.root_path_str)
        if kwargs["path_template"] is None:
            self.path_template.put(self.path_template_str)

    def stage(self, *args, **kwargs):
        self.root_path.put(self.root_path_str)
        return super().stage()

    @property
    def root_path_str(self):
        # data_session = self._redis_dict["data_session"]
        # cycle = self._redis_dict["cycle"]
        data_session = "pass-318777" # RE.md["data_session"]
        cycle = RE.md["cycle"]
        if "Commissioning" in get_proposal_type():
            root_path = f"/nsls2/data/srx/proposals/commissioning/{data_session}/assets/xspress3/"
        else:
            root_path = f"/nsls2/data/srx/proposals/{cycle}/{data_session}/assets/xspress3/"
        return root_path

    @property
    def path_template_str(self):
        path_template = "%Y/%m/%d"
        return path_template



# build a community IOC xspress3 class with 8 channels
CommunityXspress3_8Channel = build_xspress3_class(
    channel_numbers=(1, 2, 3, 4, 5, 6, 7, 8),
    mcaroi_numbers=(1, 2, 3, 4),
    image_data_key="fluor",
    xspress3_parent_classes=(Xspress3Detector, CommunitySRXXspressTrigger),
    extra_class_members={
        "hdf5": Cpt(
            Xspress3HDF5PluginWithRedis,
            "HDF1:",
            name="hdf5",
            resource_kwargs={},
            # These are overriden by properties.
            path_template=None,
            root_path=None,
        )
    }
)


class CommunitySrxXspress3Detector(CommunityXspress3_8Channel):
    # replace HDF5:FileCreateDir with HDF1:FileCreateDir
    create_dir = Cpt(EpicsSignal, "HDF1:FileCreateDir")

    # this is used as a latch to put the xspress3 into 'bulk' mode
    # for fly scanning.  Do this is a signal (rather than as a local variable
    # or as a method so we can modify this as part of a plan
    fly_next = Cpt(Signal, value=False)

    def __init__(
        self,
        prefix,
        *,
        f_key="fluor",
        configuration_attrs=None,
        read_attrs=None,
        **kwargs,
    ):
        self._f_key = f_key
        if configuration_attrs is None:
            configuration_attrs = [
                "external_trig",
                "total_points",
                "spectra_per_point",
                "cam",  # replaced settings with cam
                "rewindable",
            ]
        super().__init__(
            prefix,
            configuration_attrs=configuration_attrs,
            read_attrs=read_attrs,
            **kwargs,
        )
        if read_attrs is None:
            pass
            # JL removed channels from read_attrs
            #read_attrs = ["channel1", "channel2", "channel3", "channel4", "hdf5"]
            # JL read all rois on all channels
            #read_attrs = [
            #    xs_mcaroi.total_rbv.name
            #    for xs_channel
            #    in self.iterate_channels()
            #    for xs_mcaroi
            #    in xs_channel.iterate_mcarois()
            #]
            #read_attrs.append("hdf5")
            #print(f"read_attrs: {read_attrs}")
        # this is possiblely one too many places to store this
        # in the parent class it looks at if the extrenal_trig signal is high
        self.mode = SRXMode.fly

        # 2020-01-24
        # Commented out by AMK for using the xs3-server-IOC from TES
        # self.create_dir.put(-3)
    @property
    def mode(self):
        return self._mode

    def describe(self):
        res = super().describe()
        if self.mode == SRXMode.fly:
            res['xs_fluor']["chunks"] = (1, "auto", -1, -1)
            # Is this a "better", more future-proof way to do it?
            # This way it should automatically update the key for xs or xs2
            # and if we change the fluorescence key name, _f_key?
            # Maybe unnecessary, but I'll leave the comment in case this is helpful
            # in the future
            # res[f'{self.name}_{self._f_key}']["chunks"] = (1, "auto", -1, -1)

        return res

    @mode.setter
    def mode(self, mode):
        if not isinstance(mode, SRXMode):
            raise ValueError("mode must be of type SRXMode")
        chans = list(self.iterate_channels())
        if mode == SRXMode.step:
            self.hdf5.bulk_data_spec = 'XSP3'
            self.external_trig.put(False)
            chans[0].get_external_file_ref().kind = 1
            chans[1].get_external_file_ref().kind = 1
            chans[2].get_external_file_ref().kind = 1
            chans[3].get_external_file_ref().kind = 1
            chans[4].get_external_file_ref().kind = 1
            chans[5].get_external_file_ref().kind = 1
            chans[6].get_external_file_ref().kind = 1
            chans[7].get_external_file_ref().kind = 1
            self.get_external_file_ref().kind = 0
        if mode == SRXMode.fly:
            self.hdf5.bulk_data_spec = 'XSP3_FLY'
            self.external_trig.put(True)
            chans[0].get_external_file_ref().kind = 0
            chans[1].get_external_file_ref().kind = 0
            chans[2].get_external_file_ref().kind = 0
            chans[3].get_external_file_ref().kind = 0
            chans[4].get_external_file_ref().kind = 0
            chans[5].get_external_file_ref().kind = 0
            chans[6].get_external_file_ref().kind = 0
            chans[7].get_external_file_ref().kind = 0
            self.get_external_file_ref().kind = 1
        self._mode = mode


    def stop(self, *, success=False):
        ret = super().stop()
        # todo move this into the stop method of the settings object?
        self.cam.acquire.put(0)
        self.hdf5.stop(success=success)
        return ret

    def _compute_total_capture(self):
        total_points = self.total_points.get()
        if total_points < 1:
            raise RuntimeError("You must set the total points")
        spec_per_point = self.spectra_per_point.get()
        total_capture = total_points * spec_per_point
        return total_points, spec_per_point, total_capture

    def stage(self):
        # if should external trigger
        ext_trig = self.external_trig.get()

        # really force it to stop acquiring
        self.cam.acquire.put(0, wait=True)

        total_points, spec_per_point, total_capture = self._compute_total_capture()

        # # stop previous acquisition
        # self.stage_sigs[self.parent.cam.acquire] = 0

        # # re-order the stage signals and disable the calc record which is
        # # interfering with the capture count
        # self.stage_sigs.pop(self.num_capture, None)
        # self.stage_sigs.pop(self.parent.cam.num_images, None)
        # self.stage_sigs[self.num_capture_calc_disable] = 1

        if ext_trig:
            self.stage_sigs[self.cam.trigger_mode] = 'TTL Veto Only'
            self.stage_sigs[self.cam.num_images] = total_capture
            self.fluor.shape = (
                total_capture,
                self.hdf5.array_size_all.array_size1.get(),
                self.hdf5.array_size_all.array_size0.get(),
            )
        else:
            # self.settings.trigger_mode.put('Internal')
            # self.settings.num_images.put(1)
            self.stage_sigs[self.cam.trigger_mode] = 'Internal'
            self.stage_sigs[self.cam.num_images] = spec_per_point
            # Failed attempt to fix expected shape in tiled
            self.fluor.shape = (
                self.hdf5.array_size_all.array_size1.get(),
                self.hdf5.array_size_all.array_size0.get(),
            )

        self.stage_sigs[self.hdf5.auto_save] = 'Yes'

        # print("stage!")
        # Erase what is currently in the system
        # This prevents a single hot pixel in the upper-left corner of a map
        # JL replaced xs.erase.put(0) with self.cam.erase.put(0)
        #    why was xs.erase.put(0) not self.erase.put(0) ?
        #xs.erase.put(0)
        # JL commented out the next line because it caused a significant delay in starting acqusitions
        #self.cam.erase.put(0)
        # JL added the next line, it is not pretty

        # file_write_mode = self.hdf5.file_write_mode.get(as_string=True)
        # print(f"{file_write_mode = }")

        # self.previous_file_write_mode_value = self.hdf5.file_write_mode.get()
        # JL added the next 2 lines
        #   should use stage_sigs for file_write_mode?
        # self.hdf5.file_write_mode.put(1)
        #self.hdf5.auto_save.put(1)  # using stage_sigs for this
        # do the latching
        if self.fly_next.get():
            self.fly_next.put(False)
            self._mode = SRXMode.fly
        return super().stage()

    def unstage(self):
        # print("unstage!")
        # JL added the next two lines
        #self.hdf5.auto_save.put(0)
        # self.hdf5.file_write_mode.put(self.previous_file_write_mode_value)
        # JL removed the next line
        #self.hdf5.capture.put(0)  # this PV un-sets itself
        try:
            ret = super().unstage()
        finally:
            self._mode = SRXMode.step
        return ret


class SrxXspress3DetectorIDMonoFly(CommunitySrxXspress3Detector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._asset_docs_cache = deque()
        self._datum_counter = None
        self._datum_ids = []

    def stage(self):
        super().stage()
        self._datum_counter = itertools.count()

    def unstage(self):
        self.hdf5.capture.put(0)
        super().unstage()
        self._datum_counter = None

    def complete(self, *args, **kwargs):
        print(f'In {self.name}.complete()...')
        for resource in self.hdf5._asset_docs_cache:
            print(f'  resource in "complete": {resource}')
            self._asset_docs_cache.append(('resource', resource[1]))
        print(f'\ncomplete in {self.name}: {self._asset_docs_cache}')

        self._datum_ids = []

        # ttime.sleep(1.0)
        # num_frames = self.hdf5.num_captured.get()
        num_frames = self.cam.array_counter.get()

        print(f"{num_frames=}")
        for frame_num in range(num_frames):
            print(f'  frame_num in "complete": {frame_num + 1} / {num_frames}')
            # print(f"{self.name=}")
            for ch in xs_id_mono_fly.channel_numbers:
                print(ch)
            for ch in self.channel_numbers:
                print(ch)
            for channel in self.iterate_channels():
                print(channel)
            for channel in self.iterate_channels():
                # print(f"{self.hdf5._resource_uid=}\n{self._datum_counter=}")
                print(f"{self.hdf5._resource['uid']=}\n{self._datum_counter=}")
                datum_id = '{}/{}'.format(self.hdf5._resource['uid'], next(self._datum_counter))
                print(f"{datum_id=}")
                datum = {'resource': self.hdf5._resource['uid'],
                         'datum_kwargs': {'frame': frame_num, 'channel': channel.channel_number},
                         'datum_id': datum_id}
                print(f"{datum=}")
                self._asset_docs_cache.append(('datum', datum))
                self._datum_ids.append(datum_id)

        print(f'\nasset_docs_cache with datums:\n{self._asset_docs_cache}\n')

        return NullStatus()

    def collect(self):
        collected_frames = self.hdf5.num_captured.get()
        for frame_num in range(collected_frames):
            # print(f'  frame_num in "collect": {frame_num + 1} / {collected_frames}')

            datum_id = self._datum_ids[frame_num]
            ts = ttime.time()

            data = {self.name: datum_id}
            ts = float(ts)
            yield {'data': data,
                   'timestamps': {key: ts for key in data},
                   'time': ts,  # TODO: use the proper timestamps from the ID/mono start and stop times
                   'filled': {key: False for key in data}}

    def collect_asset_docs(self):
        items = list(self._asset_docs_cache)
        self._asset_docs_cache.clear()
        yield from items


try:
    xs_id_mono_fly = SrxXspress3DetectorIDMonoFly("XF:05IDD-ES{Xsp:3}:", name="xs_id_mono_fly")

    # add all channel.channelNN to xs.read_attrs
    read_channel_attrs = [f"channel{ch:02}" for ch in xs_id_mono_fly.channel_numbers]
    read_channel_attrs.append("hdf5")
    # print(f"read_channel_attrs: {read_channel_attrs}")
    xs_id_mono_fly.read_attrs = read_channel_attrs
    xs_id_mono_fly.fluor.dtype_str = "<u4"
    #xs_id_mono_fly.fluor.dtype_str = "<f8"
    # print(f"xs_id_mono_fly.read_attrs: {xs_id_mono_fly.read_attrs}")

    # add all mcarois.mcaroiNN.total_rbv to each channelMM.read_attrs
    for xs_channel in xs_id_mono_fly.iterate_channels():
        mcaroi_read_attrs = []
        for xs_mcaroi in xs_channel.iterate_mcarois():
            mcaroi_read_attrs = [f"mcaroi{m:02}.total_rbv" for m in range(1, 5)]
        xs_channel.read_attrs = mcaroi_read_attrs

    if os.getenv("TOUCHBEAMLINE", "0") == "1":
        print('  Touching xs_id_mono_fly...')
        # TODO add cam function
        xs_id_mono_fly.cam.num_channels.put(xs_id_mono_fly.get_channel_count())  # 4 for ME4 detector
        xs_id_mono_fly.hdf5.num_extra_dims.put(0)

        # JL replaced settings with cam
        #xs_id_mono_fly.settings.configuration_attrs = [
        xs_id_mono_fly.cam.configuration_attrs = [
            "acquire_period",
            "acquire_time",
            # "gain",
            "image_mode",
            "manufacturer",
            "model",
            "num_exposures",
            "num_images",
            "temperature",
            "temperature_actual",
            "trigger_mode",
            "config_path",
            "config_save_path",
            "invert_f0",
            "invert_veto",
            "xsp_name",
            "num_channels",
            "num_frames_config",
            "run_flags",
            "trigger_signal",
        ]

        xs_id_mono_fly.fluor.kind = "normal"
        # This is necessary for when the IOC restarts
        # We have to trigger one image for the hdf5 plugin to work correctly
        # else, we get file writing errors

        # TODO(MR): fix the warmup method:
        # xs_id_mono_fly.hdf5.warmup()
except TimeoutError as te:
    xs_id_mono_fly = None
    print(te)
    print("\nCannot connect to xs_id_mono_fly. Continuing without device.\n")
except Exception as ex:
    xs_id_mono_fly = None
    print("\nUnexpected error connecting to xs_id_mono_fly.\n")
    print(ex, end="\n\n")


try:
    print('Setting up xs...')
    xs = CommunitySrxXspress3Detector("XF:05IDD-ES{Xsp:3}:", name="xs", f_key="fluor")
    xs2 = None

    # the next line worked!
    # xs.read_attrs = ["channels.channel01.mcarois.mcaroi01.total_rbv"]
    # but putting all channels, all mcarois in one list did not work

    # add all channel.channelNN to xs.read_attrs
    read_channel_attrs = [f"channel{ch:02}" for ch in xs.channel_numbers]
    read_channel_attrs.append("hdf5")
    # print(f"read_channel_attrs: {read_channel_attrs}")
    xs.read_attrs = read_channel_attrs
    xs.fluor.dtype_str = "<u4"
    #xs.fluor.dtype_str = "<f8"
    # print(f"xs.read_attrs: {xs.read_attrs}")

    # add all mcarois.mcaroiNN.total_rbv to each channelMM.read_attrs
    for xs_channel in xs.iterate_channels():
        mcaroi_read_attrs = []
        for xs_mcaroi in xs_channel.iterate_mcarois():
            mcaroi_read_attrs = [f"mcaroi{m:02}.total_rbv" for m in range(1, 5)]
        xs_channel.read_attrs = mcaroi_read_attrs

    if os.getenv("TOUCHBEAMLINE", "0") == "1":
        print('  Touching xs...')
        xs.cam.num_channels.put(xs.get_channel_count())  # 4 for ME4 detector
        xs.hdf5.num_extra_dims.put(0)

        xs.cam.configuration_attrs = [
            "acquire_period",
            "acquire_time",
            # "gain",
            "image_mode",
            "manufacturer",
            "model",
            "num_exposures",
            "num_images",
            "temperature",
            "temperature_actual",
            "trigger_mode",
            "config_path",
            "config_save_path",
            "invert_f0",
            "invert_veto",
            "xsp_name",
            "num_channels",
            "num_frames_config",
            "run_flags",
            "trigger_signal",
        ]

        xs.fluor.kind = "normal"
        # xs.fluor.name = "fluor"  # FIXME - this causes the shape to become (251, 4, 4096)

        # TODO(MR): fix the warmup method:
        # xs.hdf5.warmup()

except TimeoutError as te:
    xs = None
    print("\nCannot connect to xs. Continuing without device.\n")
except Exception as ex:
    xs = None
    print("\nUnexpected error connecting to xs.\n")
    print(ex, end="\n\n")


def rechunk_fluor(uid):
    """
    Improve the fluor chunk shapes.
    Running this will greatly improve the data loading time.
    """
    STREAM_NAME='stream0'
    FIELD_NAME='xs_fluor'

    # Find the shape of the fluor data.
    run = tiled_reading_client[uid]
    fluor_shape = run[STREAM_NAME]["data"][FIELD_NAME].shape

    # Find which descriptors have the fluor key in them.
    descriptor_indexes = [
        index
        for index, descriptor in enumerate(run[STREAM_NAME].descriptors)
        if FIELD_NAME in descriptor["data_keys"]
    ]

    # Override the default chunking.
    for index in descriptor_indexes:
        srx_raw[uid][STREAM_NAME].patch_metadata(
            [
                {
                    "op": "add",
                    "path": f"/descriptors/{index}/data_keys/{FIELD_NAME}/chunks",
                    "value": [
                        [1] * fluor_shape[0],
                        [fluor_shape[1]],
                        [fluor_shape[2]],
                        [fluor_shape[3]],
                    ],
                }
            ]
        )
