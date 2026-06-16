print(f"Loading {__file__}...")

import time
import datetime
from ophyd import (Device, Component as Cpt,
                   EpicsSignal, EpicsSignalRO, DeviceStatus)

_time_fmtstr = '%Y-%m-%d %H:%M:%S'

from ophyd import EpicsMotor, EpicsSignal, EpicsSignalRO
from ophyd import Device
from ophyd import Component as Cpt
from ophyd import PVPositionerPC
from ophyd import Kind
from ophyd.status import WaitTimeoutError
from ophyd.utils import StatusTimeoutError
from nslsii.devices import TwoButtonShutter


# SRX-specific TwoButtonShutter
class SRXTwoButtonShutter(TwoButtonShutter):
    def stop(self, success=False):
        pass

# Setup photon shutters
shut_fe = SRXTwoButtonShutter("XF:05ID-PPS{Sh:WB}", name="shut_fe")
shut_a = SRXTwoButtonShutter("XF:05IDA-PPS:1{PSh:2}", name="shut_a")
shut_b = SRXTwoButtonShutter("XF:05IDB-PPS:1{PSh:4}", name="shut_b")

class ShutterOpeningException(Exception):
    pass

class SRXFastShutter(Device):
    NUM_RETRIES = 5
    RETRY_TIMEOUT = 1
    was_open = False
    _verbosity = 0

    open_command = Cpt(
        EpicsSignal,
        "XF:05IDD{FS:1}Open-Cmd",
        name="close_command",
        add_prefix=(),
        kind=Kind.omitted,
    )
    close_command = Cpt(
        EpicsSignal,
        "XF:05IDD{FS:1}Close-Cmd",
        name="close_command",
        add_prefix=(),
        kind=Kind.omitted,
    )
    status = Cpt(
        EpicsSignalRO,
        "XF:05IDD{FS:1}Status",
        name="status",
        add_prefix=(),
        kind=Kind.hinted,
        string=True,
    )

    def __init__(self, prefix, **kwargs):
        super().__init__(prefix, **kwargs)
        self.stage_sigs[self.open_command] = 1

    def set(self, val):
        if val not in ["Open", "Close"]:
            raise ValueError("Value can be \"Open\" or \"Close\"")

        success = False

        if val == "Open":
            val_status = "Open"
        else:
            val_status = "Closed"

        def cb(value, old_value, **kwargs):
            nonlocal val_status
            if self._verbosity > 0:
                print(f"{self.name}: {datetime.datetime.now().isoformat()} {old_value = } --> {value = }")
            if value == val_status:
                return True
            else:
                return False

        for i in range(self.NUM_RETRIES):
            try:
                st = SubscriptionStatus(self.status, callback=cb, timeout=self.RETRY_TIMEOUT, run=True)
                if val == "Open":
                    self.open_command.put(1)
                else:
                    self.close_command.put(1)
                st.wait()
                success = True
                break
            except StatusTimeoutError:
                print(f"{self.name} did not {val.lower()}! ({i+1}/{self.NUM_RETRIES})")
                continue
            except Exception as e:
                raise e

        if success is False:
            print(f"{self.name} did not {val.lower()} after {self.NUM_RETRIES} attempts!")
            raise ShutterOpeningException

        return st

    def open(self):
        return self.set("Open")

    def close(self):
        return self.set("Close")

    def stage(self):
        if self._verbosity > 0:
            banner("Opening fast shutter")
        if self.status.get() == "Open":
            self.was_open = True
        else:
            self.was_open = False
        super().stage()

    def unstage(self):
        if self._verbosity > 0:
            banner("Closing fast shutter")
        if not self.was_open:
            self.close()
        super().unstage()


shut_d = SRXFastShutter("", name="shut_d")

# Check if shutters are open
def check_shutters(check, status):
    '''
    Check if the shutters are in the desired position. At the beginning of
    a scan, they will open. At the end of the scan, they will close.

    Inputs:
    check   <bool>      Move the shutters
    status  <string>    'Open' or 'Close' Should the function be openning
                        or closing the shutters

    Returns:
     -

    '''

    if check is False:
        banner("WARNING: Shutters are not controlled in this scan.")
    else:
        if status == 'Open':
            if shut_b.status.get() == 'Not Open':
                print('Opening B-hutch shutter..')
                st = yield from mov(shut_b, "Open")
                # print(st)
                st[0].wait(10)
            print('Opening D-hutch shutter...')
            yield from mov(shut_d, "Open")
        else:
            print('Closing D-hutch shutter...')
            try:
                yield from mov(shut_d, "Close")
            except ShutterOpeningException as ex:
                print("  Error closing D-shutter!")
            except Exception as ex:
                print('  Error shutting D-shutter!')
                if 'st' in locals().keys():
                    print(st)
                print(ex)


# Setup white/pink beam slits
class SRXSlitsWB(Device):
    # Real synthetic axes
    h_cen = Cpt(EpicsMotor, "XCtr}Mtr")
    h_gap = Cpt(EpicsMotor, "XGap}Mtr")
    v_cen = Cpt(EpicsMotor, "YCtr}Mtr")
    v_gap = Cpt(EpicsMotor, "YGap}Mtr")

    # Real motors
    top = Cpt(EpicsMotor, "T}Mtr")
    bot = Cpt(EpicsMotor, "B}Mtr")
    inb = Cpt(EpicsMotor, "I}Mtr")
    out = Cpt(EpicsMotor, "O}Mtr")


class SRXSlitsPB(Device):
    # Real synthetic axes
    h_cen = Cpt(EpicsMotor, "XCtr}Mtr")
    h_gap = Cpt(EpicsMotor, "XGap}Mtr")

    # Real motors
    inb = Cpt(EpicsMotor, "I}Mtr")
    out = Cpt(EpicsMotor, "O}Mtr")


slt_wb = SRXSlitsWB("XF:05IDA-OP:1{Slt:1-Ax:", name="slt_wb")
slt_pb = SRXSlitsPB("XF:05IDA-OP:1{Slt:2-Ax:", name="slt_pb")


# Setup HFM Mirror
class SRXHFMFinePitch(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, "E-SP")  # XF:05IDA-OP:1{Mir:1-Ax:PF}E-SP
    readback = Cpt(EpicsSignalRO, "E-I")   # XF:05IDA-OP:1{Mir:1-Ax:PF}E-I

class SRXHFM(Device):
    x = Cpt(EpicsMotor, "X}Mtr")
    y = Cpt(EpicsMotor, "Y}Mtr")
    pitch = Cpt(EpicsMotor, "P}Mtr")
    fine_pitch = Cpt(SRXHFMFinePitch, "XF:05IDA-OP:1{Mir:1-Ax:PF}", name="fine_pitch", add_prefix=())
    bend = Cpt(EpicsMotor, "Bend}Mtr")


hfm = SRXHFM("XF:05IDA-OP:1{Mir:1-Ax:", name="hfm")


# Setup HDCM
class HDCMPiezoRoll(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, "")
    readback = Cpt(EpicsSignalRO, "")
    pid_enabled = Cpt(
        EpicsSignal,
        "XF:05IDD-CT{FbPid:01}PID:on",
        name="pid_enabled",
        add_prefix=()
    )
    pid_I = Cpt(
        EpicsSignal,
        "XF:05IDD-CT{FbPid:01}PID.I",
        name="pid_I",
        add_prefix=()
    )

    def reset_pid(self):
        yield from bps.mov(self.pid_I, 0.0)


class HDCMPiezoPitch(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, "")
    readback = Cpt(EpicsSignalRO, "")
    pid_enabled = Cpt(
        EpicsSignal,
        "XF:05IDD-CT{FbPid:02}PID:on",
        name="pid_enabled",
        add_prefix=()
    )
    pid_I = Cpt(
        EpicsSignal,
        "XF:05IDD-CT{FbPid:02}PID.I",
        name="pid_I",
        add_prefix=()
    )

    def reset_pid(self):
        yield from bps.mov(self.pid_I, 0.0)


class SRXDCM(Device):
    bragg = energy.bragg
    c1_roll = Cpt(EpicsMotor, "R1}Mtr")
    c1_fine = Cpt(
        HDCMPiezoRoll,
        "XF:05IDA-BI{BEST:1}PreDAC0:OutCh2", name="c1_fine",
        add_prefix=()
    )
    c2_x = energy.c2_x
    c2_pitch = Cpt(EpicsMotor, "P2}Mtr")
    c2_fine = Cpt(
        HDCMPiezoPitch,
        "XF:05IDA-BI{BEST:1}PreDAC0:OutCh1",
        name="c2_fine",
        add_prefix=()
    )
    c2_pitch_kill = Cpt(EpicsSignal, "P2}Cmd:Kill-Cmd")
    x = Cpt(EpicsMotor, "X}Mtr")
    y = Cpt(EpicsMotor, "Y}Mtr")

    temp_pitch = Cpt(EpicsSignalRO, "P}T-I")


# print('Trying to instantiate dcm from SRXDCM class...')
dcm = SRXDCM("XF:05IDA-OP:1{Mono:HDCM-Ax:", name="dcm")
dcm.wait_for_connection()
# print('Instantiated dcm from SRXDCM class!')


# Setup BPM motors
class SRXBPM(Device):
    y = Cpt(EpicsMotor, "YFoil}Mtr")
    diode_x = Cpt(EpicsMotor, "XDiode}Mtr")
    diode_y = Cpt(EpicsMotor, "YDiode}Mtr")


# These are the positioners for the backscattering diodes for bpm3/4
bpm3_pos = SRXBPM("XF:05IDA-BI:1{BPM:1-Ax:", name="bpm3_pos")
bpm4_pos = SRXBPM("XF:05IDB-BI:1{BPM:2-Ax:", name="bpm4_pos")


# Setup SSA
class SRXSSAHG(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, "X}size")
    readback = Cpt(EpicsSignalRO, "X}t2.C")


class SRXSSAHC(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, "X}center")
    readback = Cpt(EpicsSignalRO, "X}t2.D")


class SRXSSAVG(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, "Y}size")
    readback = Cpt(EpicsSignalRO, "Y}t2.C")


class SRXSSAVC(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, "Y}center")
    readback = Cpt(EpicsSignalRO, "Y}t2.D")


class SRXSSACalc(Device):
    h_cen = Cpt(SRXSSAHC, "", name="h_cen")
    h_gap = Cpt(SRXSSAHG, "", name="h_gap")
    v_cen = Cpt(SRXSSAVC, "", name="v_cen")
    v_gap = Cpt(SRXSSAVG, "", name="v_gap")


slt_ssa = SRXSSACalc("XF:05IDB-OP:1{Slt:SSA-Ax:", name="slt_ssa")


class SRXSSABLADEO(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, "xp")  # XF:05IDB-OP:1{Slt:SSA-Ax:X}xp.VAL
    readback = Cpt(EpicsSignalRO, "t2.A")  # XF:05IDB-OP:1{Slt:SSA-Ax:X}t2.A

class SRXSSABLADEI(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, "xn")
    readback = Cpt(EpicsSignalRO, "t2.B")

ssa_ob = SRXSSABLADEO("XF:05IDB-OP:1{Slt:SSA-Ax:X}", name="ssa_ob")
ssa_ib = SRXSSABLADEI("XF:05IDB-OP:1{Slt:SSA-Ax:X}", name="ssa_ib")


# Setup fast shutter
# This is not currently installed at SRX and is commented out
# class SRXSOFTINP(Device):
#     pulse = Cpt(EpicsSignal,'')
#     #these need to be put complete!!
#     def high_cmd(self):
#         self.pulse.put(1)
#     def low_cmd(self):
#         self.pulse.put(0)
#     def toggle_cmd(self):
#         if self.pulse.get() == 0:
#             self.pulse.put(1)
#         else:
#             self.pulse.put(0)
# shut_fast = SRXSOFTINP('XF:05IDD-ES:1{Sclr:1}UserLED',name='shut_fast')
#
# class SRXFASTSHUT(SRXSOFTINP):
#     pulse = Cpt(EpicsSignal,':SOFT_IN:B1')
#     iobit = Cpt(EpicsSignalRO,':SYS_STAT1LO')
#     def status(self):
#         self.low_cmd()
#         shutopen = bool(np.int16(self.iobit.get()) & np.int16(2))
#         if shutopen is True:
#             return 'Open'
#         else:
#             return 'Closed'
#     def high_cmd(self):
#         self.pulse.put(1)
#     def low_cmd(self):
#         self.pulse.put(0)
#     def open_cmd(self):
#         print(self.status())
#         if self.status() is 'Closed':
#             print(self.status())
#         #    self.low_cmd()
#             self.high_cmd()
#     def close_cmd(self):
#         print(self.status())
#         if self.status() is 'Open':
#             print(self.status())
#          #   self.low_cmd()
#             self.high_cmd()
#
# #shut_fast = SRXFASTSHUT('XF:05IDD-ES:1{Dev:Zebra1}',name='shut_fast')
