print(f"Loading {__file__}...")


import h5py
import numpy as np
from ophyd import Device, EpicsScaler, EpicsSignal, EpicsSignalRO
from ophyd import Component as Cpt
from ophyd.device import DynamicDeviceComponent as DDC
from ophyd.status import SubscriptionStatus, WaitTimeoutError
from collections import OrderedDict
from databroker.assets.handlers import HandlerBase


class EpicsSignalROLazyier(EpicsSignalRO):
    def get(self, *args, timeout=5, **kwargs):
        return super().get(*args, timeout=timeout, **kwargs)


def _scaler_fields(attr_base, field_base, range_, **kwargs):
    defn = OrderedDict()
    for i in range_:
        attr = "{attr}{i}".format(attr=attr_base, i=i)
        suffix = "{field}{i}".format(field=field_base, i=i)
        defn[attr] = (EpicsSignalROLazyier, suffix, kwargs)

    return defn

def _mca_fields(range_, **kwargs):
    defn = OrderedDict()
    for i in range_:
        mca_name = f"mca{i:02d}"
        pv_suffix = f"mca{i}"
        defn[mca_name] = (EpicsSignalRO, pv_suffix, kwargs)

    return defn


class SRXScaler(EpicsScaler):
    acquire_mode = Cpt(EpicsSignal, "AcquireMode")
    acquiring = Cpt(EpicsSignal, "Acquiring")
    asyn = Cpt(EpicsSignal, "Asyn")
    channel1_source = Cpt(EpicsSignal, "Channel1Source")
    channel_advance = Cpt(EpicsSignal, "ChannelAdvance", string=True)
    client_wait = Cpt(EpicsSignal, "ClientWait")
    count_on_start = Cpt(EpicsSignal, "CountOnStart")
    current_channel = Cpt(EpicsSignal, "CurrentChannel")
    disable_auto_count = Cpt(EpicsSignal, "DisableAutoCount")
    do_read_all = Cpt(EpicsSignal, "DoReadAll")
    dwell = Cpt(EpicsSignal, "Dwell")
    elapsed_real = Cpt(EpicsSignal, "ElapsedReal")
    enable_client_wait = Cpt(EpicsSignal, "EnableClientWait")
    erase_all = Cpt(EpicsSignal, "EraseAll")
    erase_start = Cpt(EpicsSignal, "EraseStart")
    firmware = Cpt(EpicsSignal, "Firmware")
    hardware_acquiring = Cpt(EpicsSignal, "HardwareAcquiring")
    input_mode = Cpt(EpicsSignal, "InputMode")
    max_channels = Cpt(EpicsSignal, "MaxChannels")
    model = Cpt(EpicsSignal, "Model")
    mux_output = Cpt(EpicsSignal, "MUXOutput")
    nuse_all = Cpt(EpicsSignal, "NuseAll")
    output_mode = Cpt(EpicsSignal, "OutputMode")
    output_polarity = Cpt(EpicsSignal, "OutputPolarity")
    prescale = Cpt(EpicsSignal, "Prescale")
    preset_real = Cpt(EpicsSignal, "PresetReal")
    read_all = Cpt(EpicsSignal, "ReadAll")
    read_all_once = Cpt(EpicsSignal, "ReadAllOnce")
    set_acquiring = Cpt(EpicsSignal, "SetAcquiring")
    set_client_wait = Cpt(EpicsSignal, "SetClientWait")
    snl_connected = Cpt(EpicsSignal, "SNL_Connected")
    software_channel_advance = Cpt(EpicsSignal, "SoftwareChannelAdvance")
    count_mode = Cpt(EpicsSignal, ".CONT")
    start_all = Cpt(EpicsSignal, "StartAll")
    stop_all = Cpt(EpicsSignal, "StopAll")
    user_led = Cpt(EpicsSignal, "UserLED")
    wfrm = Cpt(EpicsSignal, "Wfrm")

    _MAX_SCALER_CHANNELS = 32
    channels = DDC(_scaler_fields("chan", ".S", range(1, _MAX_SCALER_CHANNELS + 1)))
    mca_channels = DDC(_mca_fields(range(1, _MAX_SCALER_CHANNELS + 1)))


    # First channel is missing for time
    scaler_name_list = ["i0", "im", "it"] + [f"i{idx:02d}" for idx in range(5, _MAX_SCALER_CHANNELS + 1)]

    def set_num_channels(self, N):
        if N < 4 or N > 32:
            raise ValueError("N must be >= 4 and <= 32")
        
        prepended_scaler_name_list = [f"sclr_{name}" for name in self.scaler_name_list]
        
        read_attrs = []
        for index in range(N - 1):
            # Channel 1 is reserved for time. Start at 2
            read_attrs.append(f"channels.chan{index + 2}") 
            channel = getattr(self.channels, f"chan{index + 2}")
            channel.name = prepended_scaler_name_list[index]
        # for index in range(N):
        #     read_attrs.append(f"channels.chan{index + 1}")
        #     channel = getattr(self.channels, f"chan{index + 1}")
        #     channel.name = prepended_scaler_name_list[index - 1] # -1 for time channel
        self.read_attrs = read_attrs

        self._number_read_channels = N
        if hasattr(self, "_zebra_saver"):
            self._zebra_saver.sis_number_read_channels.put(N)


    def __init__(self, prefix, **kwargs):
        super().__init__(prefix, **kwargs)
        self._number_read_channels = None
    
        self.stage_sigs[self.count_mode] = "OneShot"

        # Bring mca channels up a level
        for i in range(self._MAX_SCALER_CHANNELS):
            mca_name = f"mca{i + 1:02d}"
            mca_channel = getattr(self.mca_channels, mca_name)
            mca_channel.name = f"sclr_mca_name"
            setattr(self, mca_name, mca_channel)

        self.set_num_channels(4)


sclr1 = SRXScaler("XF:05IDD-ES:1{Sclr:1}", name="sclr1")

i0 = sclr1.channels.chan2
im = sclr1.channels.chan3
it = sclr1.channels.chan4


def _zebra_saver_fields(attr_names, **kwargs):
    defn = OrderedDict()
    for name in attr_names:
        defn[name] = (EpicsSignal, name, kwargs)

    return defn

# zebra-h5-saver. Formerly file 29
class ZebraSaver(Device):
    # Saving business logic:
    write_dir = Cpt(EpicsSignal, "write_dir", string=True)
    file_name = Cpt(EpicsSignal, "file_name", string=True)
    full_file_path = Cpt(EpicsSignalRO, "full_file_path")

    acquire = Cpt(EpicsSignal, "acquire", string=True)
    file_stage = Cpt(EpicsSignal, "stage")
    dev_type = Cpt(EpicsSignal, "dev_type")

    # Zebra-related PVs:
    enc1 = Cpt(EpicsSignal, "enc1")
    enc2 = Cpt(EpicsSignal, "enc2")
    enc3 = Cpt(EpicsSignal, "enc3")
    zebra_time = Cpt(EpicsSignal, "zebra_time")

    scaler_channels = DDC(_zebra_saver_fields(["sis_time"] + sclr1.scaler_name_list))
    sis_number_read_channels = Cpt(EpicsSignal, "sis_number_read_channels")


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Bring scaler channels up a level
        for name in ["sis_time"] + sclr1.scaler_name_list:
            setattr(self, name, getattr(self.scaler_channels, name))
    

zs = ZebraSaver("XF:05IDD-ES{ZebraSaver:1}:", name="zs")
sclr1._zebra_saver = zs
sclr1.set_num_channels(4)


# Function to export scaler information
def export_sis_data(ion, filepath, zebra):

    N = ion.nuse_all.get()

    channel_attrs = {k : getattr(ion, f"mca{idx + 1:02d}")
                     for idx, k in enumerate(['time'] + ion.scaler_name_list[:ion._number_read_channels - 1])} # -1 for time channel
    channel_values = {k : attr.get(timeout=5.0) for k, attr in channel_attrs.items()}

    while len(channel_values['time']) == 0 and len(channel_values['time']) != len(channel_values['i0']):
        channel_values = {k : attr.get(timeout=5.0) for k, attr in channel_attrs.items()}

    # Count the number of points in the first channel
    if len(channel_values['i0']) != N:
        print(f"Scaler did not collect enough points.")
        # Try one more time
        channel_values = {k : attr.get(timeout=5.0) for k, attr in channel_attrs.items()}
        if len(channel_values['i0']) != N:
            print(f"Nope. Only received {len(channel_values['i0']) / N} points")
    
    # Only consider even points based on how zebra sends TTL pulses
    correct_length = N // 2
    channel_values = {k : attr[1::2] for k, attr in channel_values.items()}
    
    # Fill any missing values
    if len(channel_values['time']) != correct_length:
        correction_factor = correct_length - len(channel_values['time'])
        print(f"Adding {correction_factor} points to scaler!")
        print(f"Time is not the correct length {channel_values['time']} != {correct_length}")
        correction_list = [1e10 for _ in range(0, int(correction_factor))]
        channel_values = {k : [val for val in attr] + correction_list for k, attr in channel_values.items()}
    
    for channel, value in channel_values.items():
        if channel == "time":
            zs.sis_time.put(value.astype(int))
        else:
            getattr(zs, channel).put(value.astype(int))

    write_dir = os.path.dirname(filepath)
    file_name = os.path.basename(filepath)
    
    zs.dev_type.put("scaler")
    zs.write_dir.put(write_dir)
    zs.file_name.put(file_name)

    zs.file_stage.put("staged")

    def cb(value, old_value, **kwargs):
        import datetime
        # print(f"export_sis_data: {datetime.datetime.now().isoformat()} {old_value = } --> {value = }")
        if old_value in ["acquiring", 1] and value in ["idle", 0]:
            return True
        else:
            return False
    st = SubscriptionStatus(zs.acquire, callback=cb, run=False)
    zs.acquire.put(1)
    try:
        st.wait(timeout=60)
    except WaitTimeoutError:
        print("Scaler-save timed out! Continuing...")
    except Exception as e:
        raise e

    zs.file_stage.put("unstaged")


# # Function to export scaler information
# def export_sis_data(ion, filepath, zebra):
#     N = ion.nuse_all.get()
#     t = ion.mca01.get(timeout=5.0)
#     i0 = ion.mca02.get(timeout=5.0)
#     im = ion.mca03.get(timeout=5.0)
#     it = ion.mca04.get(timeout=5.0)
#     while len(t) == 0 and len(t) != len(i0):
#         t = ion.mca01.get(timeout=5.0)
#         i0 = ion.mca02.get(timeout=5.0)
#         im = ion.mca03.get(timeout=5.0)
#         it = ion.mca04.get(timeout=5.0)

#     if len(i0) != N:
#         print(f'Scaler did not collect enough points.')
#         ## Try one more time
#         t = ion.mca01.get(timeout=5.0)
#         i0 = ion.mca02.get(timeout=5.0)
#         im = ion.mca03.get(timeout=5.0)
#         it = ion.mca04.get(timeout=5.0)
#         if len(i) != N:
#             print(f'Nope. Only received {len(i0)}/{N} points.')

#     correct_length = N // 2
#     # correct_length = zebra.pc.data.num_down.get()
#     # Only consider even points
#     t = t[1::2]
#     i0 = i0[1::2]
#     im = im[1::2]
#     it = it[1::2]
#     # size = (len(t),)
#     # size2 = (len(i),)
#     # size3 = (len(im),)
#     # size4 = (len(it),)
#     if len(t) != correct_length:
#         correction_factor = correct_length - len(t)
#         print(f"Adding {correction_factor} points to scaler!")
#         print(f"t is not the correct length. {t} != {correct_length}")
#         correction_list = [1e10 for _ in range(0, int(correction_factor))]
#         new_t = [k for k in t] + correction_list
#         new_i0 = [k for k in i0] + correction_list
#         new_im = [k for k in im] + correction_list
#         new_it = [k for k in it] + correction_list
#     else:
#         correction_factor = 0
#         new_t = t
#         new_i0 = i0
#         new_im = im
#         new_it = it
#         # I want to define the "zero" somewhere
#         # Then if that "zero" is defined based on a 1 second count, ion chambers can be zero'ed better
#         # new = old - (zero_val * (new_t / 50_000_000))
#         # might be good to throw a np.amax(new, 0) in there to prevent negative values
#         # it would be good to save the "zero" value in the scan metadata as well
#     # with h5py.File(filepath, "w") as f:
#     #     dset0 = f.create_dataset("sis_time", (correct_length,), dtype="f")
#     #     dset0[...] = np.array(new_t)
#     #     dset1 = f.create_dataset("i0", (correct_length,), dtype="f")
#     #     dset1[...] = np.array(new_i)
#     #     dset2 = f.create_dataset("im", (correct_length,), dtype="f")
#     #     dset2[...] = np.array(new_im)
#     #     dset3 = f.create_dataset("it", (correct_length,), dtype="f")
#     #     dset3[...] = np.array(new_it)
#     #     f.close()

#     zs.i0.put(new_i0)
#     zs.im.put(new_im)
#     zs.it.put(new_it)
#     zs.sis_time.put(new_t)

#     write_dir = os.path.dirname(filepath)
#     file_name = os.path.basename(filepath)
    
#     zs.dev_type.put("scaler")
#     zs.write_dir.put(write_dir)
#     zs.file_name.put(file_name)

#     zs.file_stage.put("staged")

#     def cb(value, old_value, **kwargs):
#         import datetime
#         # print(f"export_sis_data: {datetime.datetime.now().isoformat()} {old_value = } --> {value = }")
#         if old_value in ["acquiring", 1] and value in ["idle", 0]:
#             return True
#         else:
#             return False
#     st = SubscriptionStatus(zs.acquire, callback=cb, run=False)
#     zs.acquire.put(1)
#     try:
#         st.wait(timeout=60)
#     except WaitTimeoutError:
#         print("Scaler-save timed out! Continuing...")
#     except Exception as e:
#         raise e

#     zs.file_stage.put("unstaged")


class SISHDF5Handler(HandlerBase):
    HANDLER_NAME = "SIS_HDF51"

    def __init__(self, resource_fn):
        self._handle = h5py.File(resource_fn, "r")

    def __call__(self, *, column):
        return self._handle[column][:]

    def close(self):
        self._handle.close()
        self._handle = None
        super().close()


# db.reg.register_handler("SIS_HDF51", SISHDF5Handler, overwrite=True)
