print(f"Loading {__file__}...")


import h5py
import numpy as np
from ophyd import Device, EpicsScaler, EpicsSignal, EpicsSignalRO
from ophyd import Component as Cpt
from ophyd.device import DynamicDeviceComponent as DDC
from ophyd.status import SubscriptionStatus
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


class SRXScaler(EpicsScaler):
    acquire_mode = Cpt(EpicsSignal, "AcquireMode")
    acquiring = Cpt(EpicsSignal, "Acquiring")
    asyn = Cpt(EpicsSignal, "Asyn")
    channel1_source = Cpt(EpicsSignal, "Channel1Source")
    channel_advance = Cpt(EpicsSignal, "ChannelAdvance", string=True)
    channels = DDC(_scaler_fields("chan", ".S", range(1, 33)))
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
    mca1 = Cpt(EpicsSignalRO, "mca1")
    mca2 = Cpt(EpicsSignalRO, "mca2")
    mca3 = Cpt(EpicsSignalRO, "mca3")
    mca4 = Cpt(EpicsSignalRO, "mca4")

    def __init__(self, prefix, **kwargs):
        super().__init__(prefix, **kwargs)
        self.stage_sigs[self.count_mode] = "OneShot"


sclr1 = SRXScaler("XF:05IDD-ES:1{Sclr:1}", name="sclr1")
sclr1.read_attrs = ["channels.chan2", "channels.chan3", "channels.chan4"]

i0_channel = getattr(sclr1.channels, "chan2")
i0_channel.name = "sclr_i0"
it_channel = getattr(sclr1.channels, "chan4")
it_channel.name = "sclr_it"
im_channel = getattr(sclr1.channels, "chan3")
im_channel.name = "sclr_im"

i0 = sclr1.channels.chan2
it = sclr1.channels.chan4
im = sclr1.channels.chan3


def export_sis_data(ion, filepath, zebra):
    N = ion.nuse_all.get()
    t = ion.mca1.get(timeout=5.0)
    i = ion.mca2.get(timeout=5.0)
    im = ion.mca3.get(timeout=5.0)
    it = ion.mca4.get(timeout=5.0)
    while len(t) == 0 and len(t) != len(i):
        t = ion.mca1.get(timeout=5.0)
        i = ion.mca2.get(timeout=5.0)
        im = ion.mca3.get(timeout=5.0)
        it = ion.mca4.get(timeout=5.0)

    if len(i) != N:
        print(f'Scaler did not collect enough points.')
        ## Try one more time
        t = ion.mca1.get(timeout=5.0)
        i = ion.mca2.get(timeout=5.0)
        im = ion.mca3.get(timeout=5.0)
        it = ion.mca4.get(timeout=5.0)
        if len(i) != N:
            print(f'Nope. Only received {len(i)}/{N} points.')

    correct_length = N // 2
    # correct_length = zebra.pc.data.num_down.get()
    # Only consider even points
    t = t[1::2]
    i = i[1::2]
    im = im[1::2]
    it = it[1::2]
    # size = (len(t),)
    # size2 = (len(i),)
    # size3 = (len(im),)
    # size4 = (len(it),)
    if len(t) != correct_length:
        correction_factor = correct_length - len(t)
        print(f"Adding {correction_factor} points to scaler!")
        print(f"t is not the correct length. {t} != {correct_length}")
        correction_list = [1e10 for _ in range(0, int(correction_factor))]
        new_t = [k for k in t] + correction_list
        new_i = [k for k in i] + correction_list
        new_im = [k for k in im] + correction_list
        new_it = [k for k in it] + correction_list
    else:
        correction_factor = 0
        new_t = t
        new_i = i
        new_im = im
        new_it = it
        # I want to define the "zero" somewhere
        # Then if that "zero" is defined based on a 1 second count, ion chambers can be zero'ed better
        # new = old - (zero_val * (new_t / 50_000_000))
        # might be good to throw a np.amax(new, 0) in there to prevent negative values
        # it would be good to save the "zero" value in the scan metadata as well
    # with h5py.File(filepath, "w") as f:
    #     dset0 = f.create_dataset("sis_time", (correct_length,), dtype="f")
    #     dset0[...] = np.array(new_t)
    #     dset1 = f.create_dataset("i0", (correct_length,), dtype="f")
    #     dset1[...] = np.array(new_i)
    #     dset2 = f.create_dataset("im", (correct_length,), dtype="f")
    #     dset2[...] = np.array(new_im)
    #     dset3 = f.create_dataset("it", (correct_length,), dtype="f")
    #     dset3[...] = np.array(new_it)
    #     f.close()

    zs.i0.put(new_i)
    zs.im.put(new_im)
    zs.it.put(new_it)
    zs.sis_time.put(new_t)

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
    print("Let's wait for scaler-save...", end="", flush=True)
    st.wait(timeout=60)
    print("done")
    zs.file_stage.put("unstaged")


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
