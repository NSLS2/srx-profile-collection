print(f"Loading {__file__}...")


import os
import threading
import h5py
import datetime
import numpy as np
import time as ttime
from ophyd import Device, EpicsSignal, EpicsSignalRO
from ophyd import Component as Cpt
from ophyd import FormattedComponent as FC
from ophyd.status import SubscriptionStatus
from ophyd.areadetector.filestore_mixins import FileStorePluginBase, FileStoreHDF5
# from hxntools.detectors.zebra import Zebra, EpicsSignalWithRBV
from nslsii.detectors.zebra import Zebra, EpicsSignalWithRBV


class CurrentPreampZebra(Device):
    ch0 = Cpt(EpicsSignalRO, "Cur:I0-I")
    ch1 = Cpt(EpicsSignalRO, "Cur:I1-I")
    ch2 = Cpt(EpicsSignalRO, "Cur:I2-I")
    ch3 = Cpt(EpicsSignalRO, "Cur:I3-I")

    # exp_time = Cpt(EpicsSignal, 'Per-SP')
    exp_time = Cpt(
        EpicsSignal, "XF:05IDD-ES:1{Dev:Zebra1}:PULSE3_WID", add_prefix=()
    )
    trigger_mode = Cpt(EpicsSignal, "Cmd:TrigMode")
    initi_trigger = Cpt(EpicsSignal, "Cmd:Init")
    zebra_trigger = Cpt(
        EpicsSignal, "XF:05IDD-ES:1{Dev:Zebra1}:SOFT_IN:B0", add_prefix=()
    )
    zebra_pulse_3_source = Cpt(
        EpicsSignal, "XF:05IDD-ES:1{Dev:Zebra1}:PULSE3_INP", add_prefix=()
    )

    current_scan_rate = Cpt(EpicsSignal, "Cmd:RdCur.SCAN")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs[self.zebra_trigger] = 0
        #        self.stage_sigs[self.zebra_pulse_3_source] = 44
        self.stage_sigs[self.zebra_pulse_3_source] = 60

        self.current_scan_rate.put(9)
        # update
        # self.trigger_mode.put(5)
        self.stage_sigs[self.trigger_mode] = 5  # fix this
        self.initi_trigger.put(1, wait=True)

    def stage(self):

        # Customize what is done before every scan (and undone at the end)
        # self.stage_sigs[self.trans_diode] = 5
        # or just use pyepics directly if you need to
        ret = super().stage()
        self.initi_trigger.put(1, wait=True)
        return ret

    def trigger(self):
        init_ts = self.ch0.timestamp
        timeout = float(self.exp_time.get() + 0.8)

        def retrigger():
            print(f"[WW] Re-triggered ion chamber;"
                  f"I0 for this point is suspect.")
            self.zebra_trigger.put(0, wait=True)
            self.zebra_trigger.put(1, wait=True)

        def done_cb(
            *args, obj=None, old_value=None,
            value=None, timestamp=None, **kwargs
        ):
            # if the value has changed, assume it is done
            if value != old_value:
                tmr.cancel()
                ret._finished()
                obj.clear_sub(done_cb)

        tmr = threading.Timer(timeout, retrigger)
        tmr.start()
        ret = DeviceStatus(self)

        self.ch0.subscribe(done_cb, event_type=self.ch0.SUB_VALUE, run=False)
        self.zebra_trigger.put(0, wait=True)
        self.zebra_trigger.put(1, wait=True)

        return ret


# current_preamp = CurrentPreampZebra("XF:05IDA{IM:1}", name="current_preamp")
# current_preamp = CurrentPreamp('XF:05IDA{IM:1}', name='current_preamp')


class ZebraPositionCaptureData(Device):
    """
    Data arrays for the Zebra position capture function and their metadata.
    """

    # Data arrays
    div1 = Cpt(EpicsSignal, "PC_DIV1")
    div2 = Cpt(EpicsSignal, "PC_DIV2")
    div3 = Cpt(EpicsSignal, "PC_DIV3")
    div4 = Cpt(EpicsSignal, "PC_DIV4")
    enc1 = Cpt(EpicsSignal, "PC_ENC1")
    enc2 = Cpt(EpicsSignal, "PC_ENC2")
    enc3 = Cpt(EpicsSignal, "PC_ENC3")
    enc4 = Cpt(EpicsSignal, "PC_ENC4")
    filt1 = Cpt(EpicsSignal, "PC_FILT1")
    filt2 = Cpt(EpicsSignal, "PC_FILT2")
    filt3 = Cpt(EpicsSignal, "PC_FILT3")
    filt4 = Cpt(EpicsSignal, "PC_FILT4")
    time = Cpt(EpicsSignal, "PC_TIME")
    # Array sizes
    num_cap = Cpt(EpicsSignal, "PC_NUM_CAP")
    num_down = Cpt(EpicsSignal, "PC_NUM_DOWN")
    # BOOLs to denote arrays with data
    cap_enc1_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B0")
    cap_enc2_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B1")
    cap_enc3_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B2")
    cap_enc4_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B3")
    cap_filt1_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B4")
    cap_filt2_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B5")
    cap_div1_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B6")
    cap_div2_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B7")
    cap_div3_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B8")
    cap_div4_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B9")

    def stage(self):
        super().stage()

    def unstage(self):
        super().unstage()


class ZebraPositionCapture(Device):
    """
    Signals for the position capture function of the Zebra
    """

    # Configuration settings and status PVs
    enc = Cpt(EpicsSignalWithRBV, "PC_ENC")
    egu = Cpt(EpicsSignalRO, "M1:EGU")
    dir = Cpt(EpicsSignalWithRBV, "PC_DIR")
    tspre = Cpt(EpicsSignalWithRBV, "PC_TSPRE")
    trig_source = Cpt(EpicsSignalWithRBV, "PC_ARM_SEL")
    arm = Cpt(EpicsSignal, "PC_ARM")
    disarm = Cpt(EpicsSignal, "PC_DISARM")
    armed = Cpt(EpicsSignalRO, "PC_ARM_OUT")
    gate_source = Cpt(EpicsSignalWithRBV, "PC_GATE_SEL")
    gate_start = Cpt(EpicsSignalWithRBV, "PC_GATE_START")
    gate_width = Cpt(EpicsSignalWithRBV, "PC_GATE_WID")
    gate_step = Cpt(EpicsSignalWithRBV, "PC_GATE_STEP")
    gate_num = Cpt(EpicsSignalWithRBV, "PC_GATE_NGATE")
    gated = Cpt(EpicsSignalRO, "PC_GATE_OUT")
    pulse_source = Cpt(EpicsSignalWithRBV, "PC_PULSE_SEL")
    pulse_start = Cpt(EpicsSignalWithRBV, "PC_PULSE_START")
    pulse_width = Cpt(EpicsSignalWithRBV, "PC_PULSE_WID")
    pulse_step = Cpt(EpicsSignalWithRBV, "PC_PULSE_STEP")
    pulse_max = Cpt(EpicsSignalWithRBV, "PC_PULSE_MAX")
    pulse = Cpt(EpicsSignalRO, "PC_PULSE_OUT")
    enc_pos1_sync = Cpt(EpicsSignal, "M1:SETPOS.PROC")
    enc_pos2_sync = Cpt(EpicsSignal, "M2:SETPOS.PROC")
    enc_pos3_sync = Cpt(EpicsSignal, "M3:SETPOS.PROC")
    enc_pos4_sync = Cpt(EpicsSignal, "M4:SETPOS.PROC")
    enc_res1 = Cpt(EpicsSignal, "M1:MRES")
    enc_res2 = Cpt(EpicsSignal, "M2:MRES")
    enc_res3 = Cpt(EpicsSignal, "M3:MRES")
    enc_res4 = Cpt(EpicsSignal, "M4:MRES")
    data_in_progress = Cpt(EpicsSignalRO, "ARRAY_ACQ")
    block_state_reset = Cpt(EpicsSignal, "SYS_RESET.PROC")
    data = Cpt(ZebraPositionCaptureData, "")

    def stage(self):
        self.arm.put(1)

        super().stage()

    def unstage(self):
        self.disarm.put(1)
        self.block_state_reset.put(1)

        super().unstage()


class SRXZebraOR(Device):
    # I really appreciate the different indexing for input source
    # Thank you for that
    use1 = Cpt(EpicsSignal, '_ENA:B0')
    use2 = Cpt(EpicsSignal, '_ENA:B1')
    use3 = Cpt(EpicsSignal, '_ENA:B2')
    use4 = Cpt(EpicsSignal, '_ENA:B3')
    input_source1 = Cpt(EpicsSignal, '_INP1')
    input_source2 = Cpt(EpicsSignal, '_INP2')
    input_source3 = Cpt(EpicsSignal, '_INP3')
    input_source4 = Cpt(EpicsSignal, '_INP4')
    invert1 = Cpt(EpicsSignal, '_INV:B0')
    invert2 = Cpt(EpicsSignal, '_INV:B1')
    invert3 = Cpt(EpicsSignal, '_INV:B2')
    invert4 = Cpt(EpicsSignal, '_INV:B3')

    def stage(self):
        super().stage()

    def unstage(self):
        super().unstage()


class SRXZebraAND(Device):
    # I really appreciate the different indexing for input source
    # Thank you for that
    use1 = Cpt(EpicsSignal, '_ENA:B0')
    use2 = Cpt(EpicsSignal, '_ENA:B1')
    use3 = Cpt(EpicsSignal, '_ENA:B2')
    use4 = Cpt(EpicsSignal, '_ENA:B3')
    input_source1 = Cpt(EpicsSignal, '_INP1')
    input_source2 = Cpt(EpicsSignal, '_INP2')
    input_source3 = Cpt(EpicsSignal, '_INP3')
    input_source4 = Cpt(EpicsSignal, '_INP4')
    invert1 = Cpt(EpicsSignal, '_INV:B0')
    invert2 = Cpt(EpicsSignal, '_INV:B1')
    invert3 = Cpt(EpicsSignal, '_INV:B2')
    invert4 = Cpt(EpicsSignal, '_INV:B3')

    def stage(self):
        super().stage()

    def unstage(self):
        super().unstage()



class ZebraPulse(Device):
    width = Cpt(EpicsSignalWithRBV, 'WID')
    input_addr = Cpt(EpicsSignalWithRBV, 'INP')
    input_str = Cpt(EpicsSignalRO, 'INP:STR', string=True)
    input_status = Cpt(EpicsSignalRO, 'INP:STA')
    delay = Cpt(EpicsSignalWithRBV, 'DLY')
    delay_sync = Cpt(EpicsSignal, 'DLY:SYNC')
    time_units = Cpt(EpicsSignalWithRBV, 'PRE', string=True)
    output = Cpt(EpicsSignal, 'OUT')

    input_edge = FC(EpicsSignal,
                    '{self._zebra_prefix}POLARITY:{self._edge_addr}')

    _edge_addrs = {1: 'BC',
                   2: 'BD',
                   3: 'BE',
                   4: 'BF',
                   }

    def stage(self):
        super().stage()

    def unstage(self):
        super().unstage()

    def __init__(self, prefix, *, index=None, parent=None,
                 configuration_attrs=None, read_attrs=None, **kwargs):
        if read_attrs is None:
            read_attrs = ['input_addr', 'input_edge', 'delay', 'width', 'time_units']
        if configuration_attrs is None:
            configuration_attrs = []

        zebra = parent
        self.index = index
        self._zebra_prefix = zebra.prefix
        self._edge_addr = self._edge_addrs[index]

        super().__init__(prefix, configuration_attrs=configuration_attrs,
                         read_attrs=read_attrs, parent=parent, **kwargs)



class SRXZebra(Zebra):
    """
    SRX Zebra device.
    """

    pc = Cpt(ZebraPositionCapture, "")
    or1 = Cpt(SRXZebraOR, "OR1")  # XF:05IDD-ES:1{Dev:Zebra2}:OR1_INV:B0
    or2 = Cpt(SRXZebraOR, "OR2")
    or3 = Cpt(SRXZebraOR, "OR3")
    or4 = Cpt(SRXZebraOR, "OR4")
    and1 = Cpt(SRXZebraAND, "AND1")  # XF:05IDD-ES:1{Dev:Zebra2}:AND1_ENA:B0
    and2 = Cpt(SRXZebraAND, "AND2")
    and3 = Cpt(SRXZebraAND, "AND3")
    and4 = Cpt(SRXZebraAND, "AND4")
    pulse1 = Cpt(ZebraPulse, "PULSE1_", index=1)  # XF:05IDD-ES:1{Dev:Zebra2}:PULSE1_INP
    pulse2 = Cpt(ZebraPulse, "PULSE2_", index=2)
    pulse3 = Cpt(ZebraPulse, "PULSE3_", index=3)
    pulse4 = Cpt(ZebraPulse, "PULSE4_", index=4)

    def stage(self):
        super().stage()

    def unstage(self):
        super().unstage()

    def __init__(
        self, prefix, *,
        read_attrs=None, configuration_attrs=None, **kwargs
    ):
        if read_attrs is None:
            read_attrs = []
        if configuration_attrs is None:
            configuration_attrs = []

        super().__init__(
            prefix,
            read_attrs=read_attrs,
            configuration_attrs=configuration_attrs,
            **kwargs,
        )


def set_flyer_zebra_stage_sigs(flyer, method):
    # Common stage_sigs
    ## PC Tab
    # Setup
    flyer.stage_sigs[flyer._encoder.pc.data.cap_enc1_bool] = 1
    flyer.stage_sigs[flyer._encoder.pc.data.cap_enc2_bool] = 1
    flyer.stage_sigs[flyer._encoder.pc.data.cap_enc3_bool] = 1
    flyer.stage_sigs[flyer._encoder.pc.data.cap_enc4_bool] = 0
    # flyer.stage_sigs[flyer._encoder.pc.enc] = 0
    # flyer.stage_sigs[flyer._encoder.pc.dir] = 0
    # flyer.stage_sigs[flyer._encoder.pc.tspre] = 1
    ## AND tab
    flyer.stage_sigs[flyer._encoder.and1.use1] = 1  # 0 = No, 1 = Yes
    flyer.stage_sigs[flyer._encoder.and1.use2] = 0
    flyer.stage_sigs[flyer._encoder.and1.use3] = 0
    flyer.stage_sigs[flyer._encoder.and1.use4] = 0
    flyer.stage_sigs[flyer._encoder.and1.input_source1] = 36
    flyer.stage_sigs[flyer._encoder.and1.input_source2] = 0
    flyer.stage_sigs[flyer._encoder.and1.input_source3] = 0
    flyer.stage_sigs[flyer._encoder.and1.input_source4] = 0
    flyer.stage_sigs[flyer._encoder.and1.invert1] = 1  # 0 = No, 1 = Yes
    flyer.stage_sigs[flyer._encoder.and1.invert2] = 0
    flyer.stage_sigs[flyer._encoder.and1.invert3] = 0
    flyer.stage_sigs[flyer._encoder.and1.invert4] = 0
    ## ENC tab
    flyer.stage_sigs[flyer._encoder.pc.enc_pos1_sync] = 1
    flyer.stage_sigs[flyer._encoder.pc.enc_pos2_sync] = 1
    flyer.stage_sigs[flyer._encoder.pc.enc_pos3_sync] = 1
    flyer.stage_sigs[flyer._encoder.pc.enc_pos4_sync] = 1
    ## SYS tab
    flyer.stage_sigs[flyer._encoder.output1.ttl.addr] = 31  # PC_PULSE --> TTL1 --> xs
    flyer.stage_sigs[flyer._encoder.output2.ttl.addr] = 31  # PC_PULSE --> TTL2 --> merlin
    flyer.stage_sigs[flyer._encoder.output3.ttl.addr] = 32  # OR1 --> AND1 --> TTL3 --> scaler
    flyer.stage_sigs[flyer._encoder.output4.ttl.addr] = 31  # PC_PULSE --> TTL4 --> dexela

    if method == 'position':
        flyer.mode.set('position')
        ## Specific stage sigs for Zebra - position
        # PC Tab
        # Arm
        flyer.stage_sigs[flyer._encoder.pc.trig_source] = 0
        # Gate
        flyer.stage_sigs[flyer._encoder.pc.gate_source] = 0  # 0 = Position, 1 = Time
        # flyer.stage_sigs[flyer._encoder.pc.gate_start] = 0
        # flyer.stage_sigs[flyer._encoder.pc.gate_width] = 10
        # flyer.stage_sigs[flyer._encoder.pc.gate_step] = 10.1
        flyer.stage_sigs[flyer._encoder.pc.gate_num] = 1
        # Pulse
        flyer.stage_sigs[flyer._encoder.pc.pulse_source] = 0  # 0 = Position, 1 = Time
        # flyer.stage_sigs[flyer._encoder.pc.pulse_start] = 0
        # flyer.stage_sigs[flyer._encoder.pc.pulse_width] = 0.9
        # flyer.stage_sigs[flyer._encoder.pc.pulse_step] = 1
        # flyer.stage_sigs[flyer._encoder.pc.pulse_max] = 10
        ## OR Tab
        flyer.stage_sigs[flyer._encoder.or1.use1] = 1  # 0 = No, 1 = Yes
        flyer.stage_sigs[flyer._encoder.or1.use2] = 1
        flyer.stage_sigs[flyer._encoder.or1.use3] = 1
        flyer.stage_sigs[flyer._encoder.or1.use4] = 0
        flyer.stage_sigs[flyer._encoder.or1.input_source1] = 54
        flyer.stage_sigs[flyer._encoder.or1.input_source2] = 55
        flyer.stage_sigs[flyer._encoder.or1.input_source3] = 53
        flyer.stage_sigs[flyer._encoder.or1.input_source4] = 0
        flyer.stage_sigs[flyer._encoder.or1.invert1] = 0  # 0 = No, 1 = Yes
        flyer.stage_sigs[flyer._encoder.or1.invert2] = 0
        flyer.stage_sigs[flyer._encoder.or1.invert3] = 0
        flyer.stage_sigs[flyer._encoder.or1.invert4] = 0
        ## PULSE Tab
        # flyer.stage_sigs[flyer._encoder.pulse1.input_addr] = 31
        # flyer.stage_sigs[flyer._encoder.pulse1.input_edge] = 1  # 0 = rising, 1 = falling
        # flyer.stage_sigs[flyer._encoder.pulse1.delay] = 0.2
        # flyer.stage_sigs[flyer._encoder.pulse1.width] = 0.1
        # flyer.stage_sigs[flyer._encoder.pulse1.time_units] = 0
        flyer.stage_sigs[flyer._encoder.pulse2.input_addr] = 30
        flyer.stage_sigs[flyer._encoder.pulse2.input_edge] = 0  # 0 = rising, 1 = falling
        flyer.stage_sigs[flyer._encoder.pulse2.delay] = 0
        flyer.stage_sigs[flyer._encoder.pulse2.width] = 0.1
        flyer.stage_sigs[flyer._encoder.pulse2.time_units] = 0
        flyer.stage_sigs[flyer._encoder.pulse3.input_addr] = 31
        flyer.stage_sigs[flyer._encoder.pulse3.input_edge] = 1  # 0 = rising, 1 = falling
        flyer.stage_sigs[flyer._encoder.pulse3.delay] = 0.2
        flyer.stage_sigs[flyer._encoder.pulse3.width] = 0.1
        flyer.stage_sigs[flyer._encoder.pulse3.time_units] = 0
        flyer.stage_sigs[flyer._encoder.pulse4.input_addr] = 31
        flyer.stage_sigs[flyer._encoder.pulse4.input_edge] = 1  # 0 = rising, 1 = falling
        flyer.stage_sigs[flyer._encoder.pulse4.delay] = 0
        flyer.stage_sigs[flyer._encoder.pulse4.width] = 0.1
        flyer.stage_sigs[flyer._encoder.pulse4.time_units] = 0
    elif method == 'time':
        ## Specific stage sigs for Zebra - time
        flyer.mode.set('time')
        # PC Tab
        # Arm
        flyer.stage_sigs[flyer._encoder.pc.trig_source] = 0
        # Gate
        flyer.stage_sigs[flyer._encoder.pc.gate_source] = 1  # 0 = Position, 1 = Time
        # flyer.stage_sigs[flyer._encoder.pc.gate_start] = 0
        # flyer.stage_sigs[flyer._encoder.pc.gate_width] = 10
        # flyer.stage_sigs[flyer._encoder.pc.gate_step] = 10.1
        flyer.stage_sigs[flyer._encoder.pc.gate_num] = 1
        # Pulse
        flyer.stage_sigs[flyer._encoder.pc.pulse_source] = 1  # 0 = Position, 1 = Time
        # flyer.stage_sigs[flyer._encoder.pc.pulse_start] = 0
        # flyer.stage_sigs[flyer._encoder.pc.pulse_width] = 0.9
        # flyer.stage_sigs[flyer._encoder.pc.pulse_step] = 1
        # flyer.stage_sigs[flyer._encoder.pc.pulse_max] = 10
        ## OR Tab
        flyer.stage_sigs[flyer._encoder.or1.use1] = 1  # 0 = No, 1 = Yes
        flyer.stage_sigs[flyer._encoder.or1.use2] = 1
        flyer.stage_sigs[flyer._encoder.or1.use3] = 0
        flyer.stage_sigs[flyer._encoder.or1.use4] = 0
        flyer.stage_sigs[flyer._encoder.or1.input_source1] = 54
        flyer.stage_sigs[flyer._encoder.or1.input_source2] = 55
        flyer.stage_sigs[flyer._encoder.or1.input_source3] = 53
        flyer.stage_sigs[flyer._encoder.or1.input_source4] = 0
        flyer.stage_sigs[flyer._encoder.or1.invert1] = 0  # 0 = No, 1 = Yes
        flyer.stage_sigs[flyer._encoder.or1.invert2] = 0
        flyer.stage_sigs[flyer._encoder.or1.invert3] = 0
        flyer.stage_sigs[flyer._encoder.or1.invert4] = 0
        ## PULSE Tab
        # flyer.stage_sigs[flyer._encoder.pulse1.input_addr] = 31
        # flyer.stage_sigs[flyer._encoder.pulse1.input_edge] = 1  # 0 = rising, 1 = falling
        # flyer.stage_sigs[flyer._encoder.pulse1.delay] = 0.2
        # flyer.stage_sigs[flyer._encoder.pulse1.width] = 0.1
        # flyer.stage_sigs[flyer._encoder.pulse1.time_units] = 0
        # flyer.stage_sigs[flyer._encoder.pulse2.input_addr] = 30
        # flyer.stage_sigs[flyer._encoder.pulse2.input_edge] = 0  # 0 = rising, 1 = falling
        # flyer.stage_sigs[flyer._encoder.pulse2.delay] = 0
        # flyer.stage_sigs[flyer._encoder.pulse2.width] = 0.1
        # flyer.stage_sigs[flyer._encoder.pulse2.time_units] = 0
        flyer.stage_sigs[flyer._encoder.pulse3.input_addr] = 31
        flyer.stage_sigs[flyer._encoder.pulse3.input_edge] = 0  # 0 = rising, 1 = falling
        flyer.stage_sigs[flyer._encoder.pulse3.delay] = 0.0
        flyer.stage_sigs[flyer._encoder.pulse3.width] = 0.1
        flyer.stage_sigs[flyer._encoder.pulse3.time_units] = 0
        flyer.stage_sigs[flyer._encoder.pulse4.input_addr] = 31
        flyer.stage_sigs[flyer._encoder.pulse4.input_edge] = 1  # 0 = rising, 1 = falling
        flyer.stage_sigs[flyer._encoder.pulse4.delay] = 0
        flyer.stage_sigs[flyer._encoder.pulse4.width] = 0.1
        flyer.stage_sigs[flyer._encoder.pulse4.time_units] = 0
    else:
        print('i don\'t know what to do')

class SRXFlyer1Axis(Device):
    """
    This is the flyer object for the Zebra.
    This is the position based flyer.
    """

    def root_path_str(self):
        data_session = RE.md["data_session"]
        cycle = RE.md["cycle"]
        if "Commissioning" in get_proposal_type():
            root_path = f"/nsls2/data/srx/proposals/commissioning/{data_session}/assets/"
        else:
            root_path = f"/nsls2/data/srx/proposals/{cycle}/{data_session}/assets/"
        return root_path
    

    def make_filename(self):
        """Make a filename.
        Taken/Modified from ophyd.areadetector.filestore_mixins
        This is a hook so that the read and write paths can either be modified
        or created on disk prior to configuring the areaDetector plugin.
        Returns
        -------
        filename : str
            The start of the filename
        read_path : str
            Path that ophyd can read from
        write_path : str
            Path that the IOC can write to
        """
        filename = f'{new_short_uid()}.h5'
        formatter = datetime.datetime.now().strftime
        write_path = formatter(f'{self.root_path}{self.write_path_template}')
        read_path = formatter(f'{self.root_path}{self.read_path_template}')
        return filename, read_path, write_path

    KNOWN_DETS = {"xs", "xs2", "xs4", "merlin", "dexela"}
    fast_axis = Cpt(Signal, value="HOR", kind="config")
    slow_axis = Cpt(Signal, value="VER", kind="config")
    mode = Cpt(Signal, value='position', kind='config')

    _staging_delay = 0.100  # used to be 10 ms, brute force this to work

    @property
    def encoder(self):
        return self._encoder

    @property
    def detectors(self):
        return tuple(self._dets)

    @detectors.setter
    def detectors(self, value):
        dets = tuple(value)
        if not all(d.name in self.KNOWN_DETS for d in dets):
            raise ValueError(
                f"One or more of {[d.name for d in dets]}"
                f"is not known to the zebra. "
                f"The known detectors are {self.KNOWN_DETS})"
            )
        self._dets = dets

    @property
    def sclr(self):
        return self._sis

    def __init__(self, dets, sclr1, zebra, *, reg=db.reg, **kwargs):
        super().__init__("", parent=None, **kwargs)
        self._mode = "idle"
        self._dets = dets
        self._sis = sclr1
        self._filestore_resource = None
        self._encoder = zebra

        self.root_path = self.root_path_str()
        self.write_path_template=f'zebra/%Y/%m/%d/'
        self.read_path_template=f'zebra/%Y/%m/%d/'
        self.reg_root=f'zebra/'

        # Put SIS3820 into single count (not autocount) mode
        self.stage_sigs[self._sis.count_mode] = 0

        # Stop the SIS3820
        self._sis.stop_all.put(1)

        self._encoder.pc.block_state_reset.put(1)
        self.reg = reg
        self._document_cache = []
        self._last_bulk = None

    def stage(self):
        dir = self.fast_axis.get()
        if dir == "HOR":
            self.stage_sigs[self._encoder.pc.enc] = "Enc2"
            self.stage_sigs[self._encoder.pc.dir] = "Positive"
        elif dir == "VER":
            self.stage_sigs[self._encoder.pc.enc] = "Enc1"
            self.stage_sigs[self._encoder.pc.dir] = "Positive"
        elif dir == "DET2HOR":
            self.stage_sigs[self._encoder.pc.enc] = "Enc3"
            self.stage_sigs[self._encoder.pc.dir] = "Positive"
        elif dir == "DET2VER":
            self.stage_sigs[self._encoder.pc.enc] = "Enc4"
            self.stage_sigs[self._encoder.pc.dir] = "Positive"
        elif dir == "NANOHOR":
            self.stage_sigs[self._encoder.pc.enc] = "Enc1"
            self.stage_sigs[self._encoder.pc.dir] = "Positive"
        elif dir == "NANOVER":
            self.stage_sigs[self._encoder.pc.enc] = "Enc2"
            self.stage_sigs[self._encoder.pc.dir] = "Positive"
        elif dir == "NANOZ":
            self.stage_sigs[self._encoder.pc.enc] = "Enc3"
            self.stage_sigs[self._encoder.pc.dir] = "Positive"

        self._stage_with_delay()

        self.root_path = self.root_path_str()


    def _stage_with_delay(self):
        # Staging taken from https://github.com/bluesky/ophyd/blob/master/ophyd/device.py
        # Device - BlueskyInterface
        """Stage the device for data collection.
        This method is expected to put the device into a state where
        repeated calls to :meth:`~BlueskyInterface.trigger` and
        :meth:`~BlueskyInterface.read` will 'do the right thing'.
        Staging not idempotent and should raise
        :obj:`RedundantStaging` if staged twice without an
        intermediate :meth:`~BlueskyInterface.unstage`.
        This method should be as fast as is feasible as it does not return
        a status object.
        The return value of this is a list of all of the (sub) devices
        stage, including it's self.  This is used to ensure devices
        are not staged twice by the :obj:`~bluesky.run_engine.RunEngine`.
        This is an optional method, if the device does not need
        staging behavior it should not implement `stage` (or
        `unstage`).
        Returns
        -------
        devices : list
            list including self and all child devices staged
        """
        if self._staged == Staged.no:
            pass  # to short-circuit checking individual cases
        elif self._staged == Staged.yes:
            raise RedundantStaging("Device {!r} is already staged. "
                                   "Unstage it first.".format(self))
        elif self._staged == Staged.partially:
            raise RedundantStaging("Device {!r} has been partially staged. "
                                   "Maybe the most recent unstaging "
                                   "encountered an error before finishing. "
                                   "Try unstaging again.".format(self))
        self.log.debug("Staging %s", self.name)
        self._staged = Staged.partially

        # Resolve any stage_sigs keys given as strings: 'a.b' -> self.a.b
        stage_sigs = OrderedDict()
        for k, v in self.stage_sigs.items():
            if isinstance(k, str):
                # Device.__getattr__ handles nested attr lookup
                stage_sigs[getattr(self, k)] = v
            else:
                stage_sigs[k] = v

        # Read current values, to be restored by unstage()
        original_vals = {sig: sig.get() for sig in stage_sigs}

        # We will add signals and values from original_vals to
        # self._original_vals one at a time so that
        # we can undo our partial work in the event of an error.

        # Apply settings.
        devices_staged = []
        try:
            for sig, val in stage_sigs.items():
                self.log.debug("Setting %s to %r (original value: %r)",
                               self.name,
                               val, original_vals[sig])
                sig.set(val, timeout=10).wait()
                ttime.sleep(self._staging_delay)
                # It worked -- now add it to this list of sigs to unstage.
                self._original_vals[sig] = original_vals[sig]
            ## TODO: THIS IS BAD
            ## We should fix the underlying cause, and not resend all the commands
            for sig, val in stage_sigs.items():
                self.log.debug("Setting %s to %r (original value: %r)",
                               self.name,
                               val, original_vals[sig])
                sig.set(val, timeout=10).wait()
                ttime.sleep(self._staging_delay)
            devices_staged.append(self)

            # Call stage() on child devices.
            for attr in self._sub_devices:
                device = getattr(self, attr)
                if hasattr(device, 'stage'):
                    device.stage()
                    devices_staged.append(device)
        except Exception:
            self.log.debug("An exception was raised while staging %s or "
                           "one of its children. Attempting to restore "
                           "original settings before re-raising the "
                           "exception.", self.name)
            self.unstage()
            raise
        else:
            self._staged = Staged.yes
        return devices_staged


    def unstage(self):
        self._unstage_with_delay()


    def _unstage_with_delay(self):
        # Staging taken from https://github.com/bluesky/ophyd/blob/master/ophyd/device.py
        # Device - BlueskyInterface
        """Unstage the device.
        This method returns the device to the state it was prior to the
        last `stage` call.
        This method should be as fast as feasible as it does not
        return a status object.
        This method must be idempotent, multiple calls (without a new
        call to 'stage') have no effect.
        Returns
        -------
        devices : list
            list including self and all child devices unstaged
        """
        self.log.debug("Unstaging %s", self.name)
        self._staged = Staged.partially
        devices_unstaged = []

        # Call unstage() on child devices.
        for attr in self._sub_devices[::-1]:
            device = getattr(self, attr)
            if hasattr(device, 'unstage'):
                device.unstage()
                devices_unstaged.append(device)

        # Restore original values.
        for sig, val in reversed(list(self._original_vals.items())):
            self.log.debug("Setting %s back to its original value: %r)",
                           self.name,
                           val)
            sig.set(val, timeout=10).wait()
            ttime.sleep(self._staging_delay)
            self._original_vals.pop(sig)
        devices_unstaged.append(self)

        self._staged = Staged.no
        return devices_unstaged


    def describe_collect(self):

        ext_spec = "FileStore:"

        spec = {
            "external": ext_spec,
            "dtype": "array",
            "shape": [self._npts],
            "source": "",  # make this the PV of the array the det is writing
        }

        desc = OrderedDict()
        desc["zebra_time"] = spec
        desc["zebra_time"]["source"] = getattr(self._encoder.pc.data, "time").pvname
        desc["zebra_time"]["dtype_str"] = "<f4"
        # nanoZebra.pc.data.time.dtype_str = "<f4"

        for chan in ("enc1", "enc2", "enc3"):
            desc[chan] = spec
            desc[chan]["source"] = getattr(self._encoder.pc.data, chan).pvname

        # Handle the detectors we are going to get
        for d in self._dets:
            desc.update(d.describe())

        # Handle the ion chamber that the zebra is collecting
        desc["i0"] = spec
        desc["i0"]["source"] = self._sis.mca2.pvname
        desc["i0_time"] = spec
        desc["i0_time"]["source"] = self._sis.mca1.pvname
        desc["im"] = spec
        desc["im"]["source"] = self._sis.mca3.pvname
        desc["it"] = spec
        desc["it"]["source"] = self._sis.mca4.pvname

        return {"stream0": desc}

    def kickoff(self, *, xstart, xstop, xnum, dwell, tacc):
        dets_by_name = {d.name: d for d in self.detectors}
        t_delay = 0.010  # delay after each write/put to zebra, this value is taken from _stage_with_delay

        mode = self.mode.get()
        # print(f'{mode=}')

        self._encoder.pc.arm.put(0)
        ttime.sleep(t_delay)
        self._mode = "kicked off"
        self._npts = int(xnum)
        if xstart < xstop:
            direction = 1
        else:
            direction = -1
        pxsize = np.abs(xstop - xstart) / (xnum - 1)
        extent = np.abs(xstop - xstart) + pxsize
        v = pxsize / dwell

        if mode == 'position':
            if 'dexela' in [d.name for d in self.detectors]:
                decrement = (pxsize / dwell) * 0.001
                decrement = np.max([decrement, 0.001])
            else:
                if dwell > 0.099:
                    decrement = (pxsize / dwell) * 0.001
                else:
                    decrement = (pxsize / dwell) * 0.0001
            # 0.1 ms delay between pulses
            # decrement = (pxsize / dwell) * 0.001
            if decrement < 1e-5:
                print('Warning: Changing the pulse width!')
                decrement = 1e-5
        elif mode == 'time':
            if 'dexela' in [d.name for d in self.detectors]:
                decrement = 0.001
            else:
                decrement = 0.0002
            # decrement = 0.0002

        if mode == 'position':
            self._encoder.pc.gate_start.put(xstart - direction * (pxsize / 2))
            ttime.sleep(t_delay)
            self._encoder.pc.gate_step.put(extent + 0.051)
            ttime.sleep(t_delay)
            self._encoder.pc.gate_width.put(extent + 0.050)
            ttime.sleep(t_delay)
        elif mode == 'time':
            self._encoder.pc.gate_start.put(tacc + t_delay)
            ttime.sleep(t_delay)
            # self._encoder.pc.gate_step.put(extent / v)
            self._encoder.pc.gate_step.put(extent / v + 0.100)
            ttime.sleep(t_delay)
            # self._encoder.pc.gate_width.put(extent / v + 0.050)
            self._encoder.pc.gate_width.put(extent / v)
            ttime.sleep(t_delay)


        self._encoder.pc.pulse_start.put(0.0)
        ttime.sleep(t_delay)
        self._encoder.pc.pulse_max.put(xnum)
        ttime.sleep(t_delay)
        if mode == 'position':
            self._encoder.pc.pulse_step.put(pxsize)
            ttime.sleep(t_delay)
            self._encoder.pc.pulse_width.put(pxsize - decrement)
            ttime.sleep(t_delay)
        elif mode == 'time':
            self._encoder.pc.pulse_step.put(dwell)
            ttime.sleep(t_delay)
            self._encoder.pc.pulse_width.put(dwell - decrement)
            ttime.sleep(t_delay)

        # For dexela, we will use time triggering in a pixel, not position
        # if "dexela" in dets_by_name:
        #     self._encoder.output1.ttl.addr.put(52)
        #     self._encoder.output3.ttl.addr.put(52)
        #     self._encoder.pulse1.width.put(0.5 * dwell - 0.050)
        # else:
        #     self._encoder.output1.ttl.addr.put(31)
        #     # self._encoder.output3.ttl.addr.put(31)
        #     self._encoder.output3.ttl.addr.put(36)
        #     self._encoder.pulse3.input_addr.put(31)
        #     self._encoder.pulse4.input_addr.put(31)

        self._encoder.pc.enc_pos1_sync.put(1)  # Scanner X
        ttime.sleep(t_delay)
        self._encoder.pc.enc_pos2_sync.put(1)  # Scanner Y
        ttime.sleep(t_delay)
        self._encoder.pc.enc_pos3_sync.put(1)  # Scanner Z
        ttime.sleep(t_delay)
        # self._encoder.pc.enc_pos4_sync.put(1)  # None

        # Do a block reset on the zebra
        self._encoder.pc.block_state_reset.put(1)
        ttime.sleep(t_delay)

        st = (
            NullStatus()
        )
        # TODO Return a status object *first*
        # and do the above asynchronously.
        return st

    def complete(self):
        """
        Call this when all needed data has been collected. This has no idea
        whether that is true, so it will obligingly stop immediately. It is
        up to the caller to ensure that the motion is actually complete.
        """

        amk_debug_flag = False

        # Our acquisition complete PV is: XF:05IDD-ES:1{Dev:Zebra1}:ARRAY_ACQ
        t0 = ttime.monotonic()
        while self._encoder.pc.data_in_progress.get() == 1:
            ttime.sleep(0.01)
            if (ttime.monotonic() - t0) > 60:
                print(f"{self.name} is behaving badly!")
                print(f"Number of points captured:   {self._encoder.pc.data.num_cap.get()}")
                print(f"Number of points downloaded: {self._encoder.pc.data.num_down.get()}")
                self._encoder.pc.disarm.put(1)
                ttime.sleep(0.100)
                if self._encoder.pc.data_in_progress.get() == 1:
                    print("Last row of data is lost :-(")
                    print("Aborting scan")
                    raise TimeoutError

        # ttime.sleep(.1)
        self._mode = "complete"
        self._encoder.pc.block_state_reset.put(1)
        # see triggering errors of the xspress3 on suspension.  This is
        # to test the reset of the xspress3 after a line.

        for d in self._dets:
            d.stop(success=True)

        # Set filename/path for zebra data
        f, rp, wp = self.make_filename()
        self.__filename = f
        self.__read_filepath = os.path.join(rp, self.__filename)
        self.__write_filepath = os.path.join(wp, self.__filename)
        # Set filename/path for scaler data
        f, rp, wp = self.make_filename()
        self.__filename_sis = f
        self.__read_filepath_sis = os.path.join(rp, self.__filename_sis)
        self.__write_filepath_sis = os.path.join(wp, self.__filename_sis)
        

        self.__filestore_resource, datum_factory_z = resource_factory(
            "ZEBRA_HDF51",
            root="/",
            resource_path=self.__read_filepath,
            resource_kwargs={},
            path_semantics="posix",
        )
        self.__filestore_resource_sis, datum_factory_sis = resource_factory(
            "SIS_HDF51",
            root="/",
            resource_path=self.__read_filepath_sis,
            resource_kwargs={},
            path_semantics="posix",
        )

        time_datum = datum_factory_z({"column": "zebra_time"})
        enc1_datum = datum_factory_z({"column": "enc1"})
        enc2_datum = datum_factory_z({"column": "enc2"})
        enc3_datum = datum_factory_z({"column": "enc3"})
        sis_datum = datum_factory_sis({"column": "i0"})
        sis_datum_im = datum_factory_sis({"column": "im"})
        sis_datum_it = datum_factory_sis({"column": "it"})
        sis_datum_time = datum_factory_sis({"column": "sis_time"})

        self._document_cache.extend(
            ("resource", d)
            for d in (self.__filestore_resource, self.__filestore_resource_sis)
        )
        self._document_cache.extend(
            ("datum", d)
            for d in (
                time_datum,
                enc1_datum,
                enc2_datum,
                enc3_datum,
                sis_datum,
                sis_datum_time,
                sis_datum_im,
                sis_datum_it,
            )
        )

        # grab the asset documents from all of the child detectors
        for d in self._dets:
            self._document_cache.extend(d.collect_asset_docs())

        # Write the file.
        # @timer_wrapper
        def get_zebra_data():
            if 'nano' in self.name:
                export_nano_zebra_data(self._encoder, self.__write_filepath, self.fast_axis.get())
            else:
                export_zebra_data(self._encoder, self.__write_filepath, self.fast_axis)

        if amk_debug_flag:
            t_getzebradata = tic()
        get_zebra_data()
        if amk_debug_flag:
            toc(t_getzebradata, str='Get Zebra data')

        # @timer_wrapper
        def get_sis_data():
            export_sis_data(
                self._sis, self.__write_filepath_sis, self._encoder
            )

        if amk_debug_flag:
            t_sisdata = tic()
        get_sis_data()
        if amk_debug_flag:
            toc(t_sisdata, str='Get SIS data')

        # Yield a (partial) Event document. The RunEngine will put this
        # into metadatastore, as it does all readings.
        self._last_bulk = {
            "time": ttime.time(),
            "seq_num": 1,
            "data": {
                "zebra_time": time_datum["datum_id"],
                "enc1": enc1_datum["datum_id"],
                "enc2": enc2_datum["datum_id"],
                "enc3": enc3_datum["datum_id"],
                "i0": sis_datum["datum_id"],
                "i0_time": sis_datum_time["datum_id"],
                "im": sis_datum_im["datum_id"],
                "it": sis_datum_it["datum_id"],
            },
            "timestamps": {
                "zebra_time": time_datum["datum_id"],  # not a typo#
                "enc1": time_datum["datum_id"],
                "enc2": time_datum["datum_id"],
                "enc3": time_datum["datum_id"],
                "i0": sis_datum["datum_id"],
                "i0_time": sis_datum_time["datum_id"],
                "im": sis_datum_im["datum_id"],
                "it": sis_datum_it["datum_id"],
            },
        }
        for d in self._dets:
            reading = d.read()
            self._last_bulk["data"].update(
                {k: v["value"] for k, v in reading.items()}
                )
            self._last_bulk["timestamps"].update(
                {k: v["timestamp"] for k, v in reading.items()}
            )

        return NullStatus()

    def collect(self):
        # Create records in the FileStore database.
        # move this to stage because I thinkt hat describe_collect needs the
        # resource id
        # TODO use ophyd.areadectector.filestoer_mixins.resllource_factory here
        if self._last_bulk is None:
            raise Exception(
                "the order of complete and collect is brittle and out "
                "of sync. This device relies on in-order and 1:1 calls "
                "between complete and collect to correctly create and stash "
                "the asset registry documents"
            )
        yield self._last_bulk
        self._last_bulk = None
        self._mode = "idle"

    def collect_asset_docs(self):
        yield from iter(list(self._document_cache))
        self._document_cache.clear()

    def stop(self):
        self._encoder.pc.block_state_reset.put(1)
        pass

    def pause(self):
        "Pausing in the middle of a kickoff nukes the partial dataset."
        self._encoder.pc.block_state_reset.put(1)
        self._sis.stop_all.put(1)
        for d in self._dets:
            if hasattr(d, "settings"):
                d.settings.acquire.put(0)
            if hasattr(d, "cam"):
                d.cam.acquire.put(0)
        self._mode = "idle"
        self.unstage()

    def resume(self):
        self.unstage()
        self.stage()

# For microES
try:
    microZebra = SRXZebra("XF:05IDD-ES:1{Dev:Zebra1}:", name="microZebra",
        read_attrs=["pc.data.enc1", "pc.data.enc2", "pc.data.time"],
    )
    # There is no flyer attached to Zebra1.
    # Leaving object as none until it is confirmed that it can be removed.
    flying_zebra = None
    # flying_zebra = SRXFlyer1Axis(
    #     list(xs for xs in [xs] if xs is not None), sclr1, microZebra, name="flying_zebra"
    # )
except Exception as ex:
    print("Cannot connect to microZebra. Continuing without device.\n", ex)
    raise ex
    flying_zebra = None


# For nanoES
try:
    # Setup nanoZebra
    nanoZebra = SRXZebra("XF:05IDD-ES:1{Dev:Zebra2}:", name="nanoZebra",
        read_attrs=["pc.data.enc1", "pc.data.enc2", "pc.data.enc3", "pc.data.time"],
    )
    
    if os.getenv("TOUCHBEAMLINE", "0") == "1":
        print('  Touching nanoZebra...', end='')
        # Set encoder resolution on startup
        nanoZebra.pc.enc_res1.put(-9.5368e-05)
        nanoZebra.pc.enc_res2.put(9.5368e-05)
        nanoZebra.pc.enc_res3.put(9.5368e-05)
        print('done')

    nano_flying_zebra = SRXFlyer1Axis(
        list(xs for xs in [xs] if xs is not None), sclr1, nanoZebra, name="nano_flying_zebra"
    )
    set_flyer_zebra_stage_sigs(nano_flying_zebra, 'position')

    nano_flying_zebra_coarse = SRXFlyer1Axis(
        list(xs for xs in [xs] if xs is not None), sclr1, nanoZebra, name="nano_flying_zebra_coarse"
    )
    set_flyer_zebra_stage_sigs(nano_flying_zebra_coarse, 'time')

    # Temporary flyer for ME4 until ME7 is commissioned
    # nano_flying_zebra_me4 = SRXFlyer1Axis(
    #     list(xs4 for xs4 in [xs4] if xs4 is not None), sclr1, nanoZebra, name="nano_flying_zebra_me4"
    # )
    # set_flyer_zebra_stage_sigs(nano_flying_zebra_me4, 'position')
    nano_flying_zebra_me4 = None

    # Temporary flyer for ME4 until ME7 is commissioned
    # nano_flying_zebra_coarse_me4 = SRXFlyer1Axis(
    #     list(xs4 for xs4 in [xs4] if xs4 is not None), sclr1, nanoZebra, name="nano_flying_zebra_coarse_me4"
    # )
    # set_flyer_zebra_stage_sigs(nano_flying_zebra_coarse_me4, 'time')
    nano_flying_zebra_coarse_me4 = None
except Exception as ex:
    print("Cannot connect to nanoZebra. Continuing without device.\n", ex)
    nano_flying_zebra = None
    nano_flying_zebra_coarse = None
    nano_flying_zebra_me4 = None


# For confocal
# For plans that call xs2,
# should we simply add xs2 to flying_zebra.dets
# and set dir to 'DET2HOR'?
# if xs2 is not None:
#     # flying_zebra_x_xs2 = SRXFlyer1Axis(
#     #     zebra, [xs2], sclr1, "HOR", name="flying_zebra_x_xs2"
#     # )
#     # flying_zebra_y_xs2 = SRXFlyer1Axis(
#     #     zebra, [xs2], sclr1, "VER", name="flying_zebra_y_xs2"
#     # )
#     flying_zebra_xs2 = SRXFlyer1Axis(
#         list(xs2 for xs2 in [xs2] if xs2 is not None),
#         sclr1,
#         nanoZebra,
#         name="flying_zebra_xs2"
#     )
#
# else:
#     flying_zebra_xs2 = None
#     # flying_zebra_y_xs2 = None
# For chip imaging
# flying_zebra_x_xs2 = SRXFlyer1Axis(
#   zebra, xs2, sclr1, 'DET2HOR', name='flying_zebra'
# )
# flying_zebra_y_xs2 = SRXFlyer1Axis(
#   zebra, xs2, sclr1, 'DET2VER', name='flying_zebra'
# )
# flying_zebra = SRXFlyer1Axis(zebra)


def export_nano_zebra_data(zebra, filepath, fastaxis):
    j = 0
    while zebra.pc.data_in_progress.get() == 1:
        print("Waiting for zebra...")
        ttime.sleep(0.1)
        j += 1
        if j > 10:
            print("THE ZEBRA IS BEHAVING BADLY CARRYING ON")
            break

    time_d = zebra.pc.data.time.get()
    enc1_d = zebra.pc.data.enc1.get()
    enc2_d = zebra.pc.data.enc2.get()
    enc3_d = zebra.pc.data.enc3.get()

    px = zebra.pc.pulse_step.get()
    if fastaxis == 'NANOHOR':
        # Add half pixelsize to correct encoder
        enc1_d = enc1_d + (px / 2)
    elif fastaxis == 'NANOVER':
        # Add half pixelsize to correct encoder
        enc2_d = enc2_d + (px / 2)
    elif fastaxis == 'NANOZ':
        # Add half pixelsize to correct encoder
        enc3_d = enc3_d + (px / 2)

    # size = (len(time_d),)
    # with h5py.File(filepath, "w") as f:
    #     dset0 = f.create_dataset("zebra_time", size, dtype="f")
    #     dset0[...] = np.array(time_d)
    #     dset1 = f.create_dataset("enc1", size, dtype="f")
    #     dset1[...] = np.array(enc1_d)
    #     dset2 = f.create_dataset("enc2", size, dtype="f")
    #     dset2[...] = np.array(enc2_d)
    #     dset3 = f.create_dataset("enc3", size, dtype="f")
    #     dset3[...] = np.array(enc3_d)

    zs.enc1.put(enc1_d)
    zs.enc2.put(enc2_d)
    zs.enc3.put(enc3_d)
    zs.zebra_time.put(time_d)

    write_dir = os.path.dirname(filepath)
    file_name = os.path.basename(filepath)
    
    zs.dev_type.put("zebra")
    zs.write_dir.put(write_dir)
    zs.file_name.put(file_name)

    zs.file_stage.put("staged")

    def cb(value, old_value, **kwargs):
        import datetime
        # print(f"export_nano_zebra_data: {datetime.datetime.now().isoformat()} {old_value = } --> {value = }")
        if old_value in ["acquiring", 1] and value in ["idle", 0]:
            return True
        else:
            return False
    st = SubscriptionStatus(zs.acquire, callback=cb, run=False)
    zs.acquire.put(1)
    st.wait()

    zs.file_stage.put("unstaged")


def export_zebra_data(zebra, filepath, fast_axis):
    print('\n\n\nI am in micro export\n\n\n\n')
    j = 0
    while zebra.pc.data_in_progress.get() == 1:
        print("waiting zebra")
        ttime.sleep(0.1)
        j += 1
        if j > 10:
            print("THE ZEBRA IS BEHAVING BADLY CARRYING ON")
            break

    time_d = zebra.pc.data.time.get()
    if fast_axis == "HOR":
        enc1_d = zebra.pc.data.enc2.get()
        enc2_d = zebra.pc.data.enc1.get()
    elif fast_axis == "DET2HOR":
        enc1_d = zebra.pc.data.enc3.get()
    elif fast_axis == "DET2VER":
        enc1_d = zebra.pc.data.enc4.get()
    else:
        enc1_d = zebra.pc.data.enc1.get()
        enc2_d = zebra.pc.data.enc2.get()

    enc3_d = 0*enc2_d

    while len(time_d) == 0 or len(time_d) != len(enc1_d):
        time_d = zebra.pc.data.time.get()
        if fast_axis == "HOR":
            enc1_d = zebra.pc.data.enc2.get()
        else:
            enc1_d = zebra.pc.data.enc1.get()

    size = (len(time_d),)
    with h5py.File(filepath, "w") as f:
        dset0 = f.create_dataset("time", size, dtype="f")
        dset0[...] = np.array(time_d)
        dset1 = f.create_dataset("enc1", size, dtype="f")
        dset1[...] = np.array(enc1_d)
        dset2 = f.create_dataset("enc2", size, dtype="f")
        dset2[...] = np.array(enc2_d)
        dset3 = f.create_dataset("enc3", size, dtype="f")
        dset3[...] = np.array(enc3_d)


class ZebraHDF5Handler(HandlerBase):
    HANDLER_NAME = "ZEBRA_HDF51"

    def __init__(self, resource_fn):
        self._handle = h5py.File(resource_fn, "r")

    def __call__(self, *, column):
        return self._handle[column][:]

    def close(self):
        self._handle.close()
        self._handle = None
        super().close()


# db.reg.register_handler("ZEBRA_HDF51", ZebraHDF5Handler, overwrite=True)
