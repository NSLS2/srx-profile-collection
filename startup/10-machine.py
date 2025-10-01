print(f"Loading {__file__}...")

import numpy as np
from ophyd import (
    EpicsSignal,
    EpicsSignalRO,
    EpicsMotor,
    Device,
    Signal,
    PseudoPositioner,
    PseudoSingle,
)
from ophyd.utils import ReadOnlyError
from ophyd.pseudopos import pseudo_position_argument, real_position_argument
from ophyd.positioner import PositionerBase
from ophyd import Component as Cpt
from ophyd.status import SubscriptionStatus

from scipy.interpolate import make_interp_spline
import functools
import math
from pathlib import Path


"""
For organization, this file will define objects for the machine. This will
include the undulator (and energy axis) and front end slits.
"""


# Constants
ANG_OVER_EV = 12.3984


# Signals
ring_current = EpicsSignalRO("SR:C03-BI{DCCT:1}I:Real-I", name="ring_current")


# Setup undulator
class InsertionDevice(Device, PositionerBase):
    gap = Cpt(EpicsMotor, "-Ax:Gap}-Mtr", kind="hinted", name="")
    brake = Cpt(
        EpicsSignal,
        "}BrakesDisengaged-Sts",
        write_pv="}BrakesDisengaged-SP",
        kind="omitted",
        add_prefix=("read_pv", "write_pv", "suffix"),
    )

    # These are debugging values, not even connected to by default
    elev = Cpt(EpicsSignalRO, "-Ax:Elev}-Mtr.RBV", kind="omitted")
    taper = Cpt(EpicsSignalRO, "-Ax:Taper}-Mtr.RBV", kind="omitted")
    tilt = Cpt(EpicsSignalRO, "-Ax:Tilt}-Mtr.RBV", kind="omitted")
    elev_u = Cpt(EpicsSignalRO, "-Ax:E}-Mtr.RBV", kind="omitted")

    def set(self, *args, **kwargs):
        # set_and_wait(self.brake, 1) // deprecated
        self.brake.set(1).wait()
        return self.gap.set(*args, **kwargs)

    def stop(self, *, success=False):
        return self.gap.stop(success=success)

    @property
    def settle_time(self):
        return self.gap.settle_time

    @settle_time.setter
    def settle_time(self, val):
        self.gap.settle_time = val

    @property
    def timeout(self):
        return self.gap.timeout

    @timeout.setter
    def timeout(self, val):
        self.gap.timeout = val

    @property
    def egu(self):
        return self.gap.egu

    @property
    def limits(self):
        return self.gap.limits

    @property
    def low_limit(self):
        return self.gap.low_limit

    @property
    def high_limit(self):
        return self.gap.high_limit

    def move(self, *args, moved_cb=None, **kwargs):
        if moved_cb is not None:

            @functools.wraps(moved_cb)
            def inner_move(status, obj=None):
                if obj is not None:
                    obj = self
                return moved_cb(status, obj=obj)

        else:
            inner_move = None
        return self.set(*args, moved_cb=inner_move, **kwargs)

    @property
    def position(self):
        return self.gap.position

    @property
    def moving(self):
        return self.gap.moving

    def subscribe(self, callback, *args, **kwargs):
        @functools.wraps(callback)
        def inner(obj, **kwargs):
            return callback(obj=self, **kwargs)

        return self.gap.subscribe(inner, *args, **kwargs)


# Setup energy axis
class Energy(PseudoPositioner):
    # Synthetic axis
    energy = Cpt(PseudoSingle, egu='keV')
    egu = Cpt(Signal, None, add_prefix=(), value='keV', kind="config")

    # Real motors
    u_gap = Cpt(InsertionDevice, "SR:C5-ID:G1{IVU21:1")
    _u_gap_offset = 0
    bragg = Cpt(
        EpicsMotor,
        "XF:05IDA-OP:1{Mono:HDCM-Ax:P}Mtr",
        add_prefix=(),
        read_attrs=["user_readback"],
    )
    c2_x = Cpt(
        EpicsMotor,
        "XF:05IDA-OP:1{Mono:HDCM-Ax:X2}Mtr",
        add_prefix=(),
        read_attrs=["user_readback"],
    )
    epics_d_spacing = EpicsSignal("XF:05IDA-CT{IOC:Status01}DCMDspacing.VAL")
    epics_bragg_offset = EpicsSignal("XF:05IDA-CT{IOC:Status01}BraggOffset.VAL")

    # Energy "limits"
    _low = 4.4
    _high = 25

    # Motor enable flags
    move_u_gap = Cpt(Signal, None, add_prefix=(), value=True)
    move_c2_x = Cpt(Signal, None, add_prefix=(), value=True)
    harmonic = Cpt(Signal, None, add_prefix=(), value=0, kind="config")
    selected_harmonic = Cpt(Signal, None, add_prefix=(), value=0)

    # Experimental
    detune = Cpt(Signal, None, add_prefix=(), value=0)

    def energy_to_positions(self, target_energy, undulator_harmonic, u_detune):
        """Compute undulator and mono positions given a target energy

        Paramaters
        ----------
        target_energy : float
            Target energy in keV

        undulator_harmonic : int, optional
            The harmonic in the undulator to use

        uv_mistune : float, optional
            Amount to 'mistune' the undulator in keV.  Will settings
            such that the peak of the undulator spectrum will be at
            `target_energy + uv_mistune`.

        Returns
        -------
        bragg : float
            The angle to set the monocromotor
        c2x : float
            The c2x position
        ugap : float
            The undulator gap in microns

        """
        # Set up constants
        Xoffset = self._xoffset
        d_111 = self._d_111
        delta_bragg = self._delta_bragg
        C2Xcal = self._c2xcal
        T2cal = self._t2cal
        etoulookup = self.etoulookup

        # Calculate Bragg RBV
        BraggRBV = (
            np.arcsin((ANG_OVER_EV / target_energy) / (2 * d_111)) / np.pi * 180
            - delta_bragg
        )

        # Calculate C2X
        Bragg = BraggRBV + delta_bragg
        T2 = Xoffset * np.sin(Bragg * np.pi / 180) / np.sin(2 * Bragg * np.pi / 180)
        dT2 = T2 - T2cal
        C2X = C2Xcal - dT2

        # Calculate undulator gap

        #  TODO make this more sohpisticated to stay a fixed distance
        #  off the peak of the undulator energy
        ugap = float(
            etoulookup((target_energy + u_detune) / undulator_harmonic)
        )  # in mm
        ugap *= 1000  # convert to um
        ugap = ugap + self._u_gap_offset

        return BraggRBV, C2X, ugap

    def undulator_energy(self, harmonic=3):
        """Return the current energy peak of the undulator at the given harmonic

        Paramaters
        ----------
        harmonic : int, optional
            The harmonic to use, defaults to 3
        """
        ugap = self.u_gap.gap.user_readback.get() / 1000
        utoelookup = self.utoelookup

        fundemental = float(utoelookup(ugap))

        energy = fundemental * harmonic

        return energy

    def __init__(
        self,
        *args,
        xoffset=None,
        d_111=None,
        delta_bragg=None,
        C2Xcal=None,
        T2cal=None,
        **kwargs,
    ):
        self._xoffset = xoffset
        self._d_111 = d_111
        self._delta_bragg = delta_bragg
        self._c2xcal = C2Xcal
        self._t2cal = T2cal
        super().__init__(*args, **kwargs)

        calib_path = Path(__file__).parent
        calib_file = "../data/SRXUgapCalibration.txt"  # /nsls2/data/srx/shared/config/bluesky/profile_collection/data 

        # with open(os.path.join(calib_path, calib_file), 'r') as f:
        with open(calib_path / calib_file, "r") as f:
            next(f)
            uposlistIn = []
            elistIn = []
            for line in f:
                num = [float(x) for x in line.split()]
                # Check in case there is an extra line at the end of the calibration file
                if len(num) == 2:
                    uposlistIn.append(num[0])
                    elistIn.append(num[1])

        self.etoulookup = make_interp_spline(elistIn, uposlistIn)
        self.utoelookup = make_interp_spline(uposlistIn, elistIn)

        self.u_gap.gap.user_readback.name = self.u_gap.name

    def crystal_gap(self):
        """
        Return the current physical gap between first and second crystals
        """
        C2X = self.c2_x.get().user_readback
        bragg = self.bragg.get().user_readback

        T2cal = self._t2cal
        delta_bragg = self._delta_bragg
        d_111 = self._d_111
        c2x_cal = self._c2xcal

        Bragg = np.pi / 180 * (bragg + delta_bragg)

        dT2 = c2x_cal - C2X
        T2 = dT2 + T2cal

        XoffsetVal = T2 / (np.sin(Bragg) / np.sin(2 * Bragg))

        return XoffsetVal

    @pseudo_position_argument
    def forward(self, p_pos):
        energy = p_pos.energy
        harmonic = int(self.harmonic.get())
        if harmonic < 0 or ((harmonic % 2) == 0 and harmonic != 0):
            raise RuntimeError(
                f"The harmonic must be 0 or odd and positive, you set {harmonic}.  "
                "Set `energy.harmonic` to a positive odd integer or 0."
            )
        detune = self.detune.get()
        if energy <= self._low:
            raise ValueError(
                f"The energy you entered is too low ({energy} keV). "
                f"Minimum energy = {self._low:.1f} keV"
            )
        if energy > self._high:
            if (energy < self._low * 1000) or (energy > self._high * 1000):
                # Energy is invalid
                raise ValueError(
                    f"The requested photon energy is invalid ({energy} keV). "
                    f"Values must be in the range of {self._low:.1f} - {self._high:.1f} keV"
                )
            else:
                # Energy is in eV
                energy = energy / 1000.0

        # harmonic cannot be None, it is an undesired datatype
        # Previously, we were finding the harmonic with the highest flux, this
        # was always done during energy change since harmonic was returned to
        # None
        # Here, we are programming it in
        # if harmonic is None:
        if harmonic < 3:
            harmonic = 3
            # Choose the right harmonic
            braggcal, c2xcal, ugapcal = self.energy_to_positions(
                energy, harmonic, detune
            )
            # Try higher harmonics until the required gap is too small
            while True:
                braggcal, c2xcal, ugapcal = self.energy_to_positions(
                    energy, harmonic + 2, detune
                )
                if ugapcal < self.u_gap.low_limit:
                    break
                harmonic += 2

        self.selected_harmonic.put(harmonic)

        # Compute where we would move everything to in a perfect world
        bragg, c2_x, u_gap = self.energy_to_positions(energy, harmonic, detune)

        # Sometimes move the crystal gap
        if not self.move_c2_x.get():
            c2_x = self.c2_x.position

        # Sometimes move the undulator
        if not self.move_u_gap.get():
            u_gap = self.u_gap.position

        return self.RealPosition(bragg=bragg, c2_x=c2_x, u_gap=u_gap)

    @real_position_argument
    def inverse(self, r_pos):
        bragg = r_pos.bragg
        e = ANG_OVER_EV / (
            2 * self._d_111 * math.sin(math.radians(bragg + self._delta_bragg))
        )
        return self.PseudoPosition(energy=float(e))

    @pseudo_position_argument
    def set(self, position):
        return super().set([float(_) for _ in position])

    def synch_with_epics(self):
        self.epics_d_spacing.put(self._d_111)
        self.epics_bragg_offset.put(self._delta_bragg)

    def retune_undulator(self):
        self.detune.put(0.0)
        self.move(self.energy.get()[0])


cal_data_2025cycle3 = {
    "d_111": 3.1288110559600995,
    "delta_bragg": 0.23220073137275954,
    "C2Xcal": 3.6,
    "T2cal": 15.0347755916,
    "xoffset": 24.65,
}

# print('Connecting to energy PVs...')
energy = Energy(prefix="", name="energy", **cal_data_2025cycle3)
energy.wait_for_connection()
energy.synch_with_epics()
energy.value = 1.0



# Setup front end slits (primary slits)
class SRXSlitsFE(Device):
    top = Cpt(EpicsMotor, "3-Ax:T}Mtr")
    bot = Cpt(EpicsMotor, "4-Ax:B}Mtr")
    inb = Cpt(EpicsMotor, "3-Ax:I}Mtr")
    out = Cpt(EpicsMotor, "4-Ax:O}Mtr")

# print('Connecting to FE slit PVs...')
fe = SRXSlitsFE("FE:C05A-OP{Slt:", name="fe")


class FlyScanControl(Device):
    control = Cpt(EpicsSignal, write_pv='MACROControl-SP', read_pv='MACROControl-RB',
                  add_prefix=('read_pv', 'write_pv'), put_complete=True)
    status = Cpt(EpicsSignalRO, 'MACRO-Sts')
    reset = Cpt(EpicsSignal, 'MACRO-CLRF.PROC')

    def set(self, command):
        allowed_commands = {"enable", "disable"}
        ENABLED_VALUE = 5
        DISABLED_VALUE = 10

        def _int_round(value):
            return int(round(value))

        if command == "enable":
            def enable_callback(value, old_value, **kwargs):
                # print(f'{print_now()} in {self.name}/{command}: {old_value} ---> {value}')
                value = _int_round(value)
                old_value = _int_round(old_value)
                if value == ENABLED_VALUE:
                    return True
                return False
            status = SubscriptionStatus(self.control, enable_callback, run=False)
            self.control.put(1)
            return status

        elif command == "disable":
            def disable_callback(value, old_value, **kwargs):
                # print(f'{print_now()} in {self.name}/{command}: {old_value} ---> {value}')
                value = _int_round(value)
                old_value = _int_round(old_value)
                if value == DISABLED_VALUE:
                    return True
                return False
            status = SubscriptionStatus(self.control, disable_callback, run=False)
            self.control.put(0)
            return status
        else:
            raise ValueError(f"Unknown command: {command}. "
                             f"Allowed commands: {allowed_commands}")

    scan_type = Cpt(EpicsSignal, write_pv='FlyScan-Type-SP', read_pv='FlyScan-Type-RB', add_prefix=('read_pv', 'write_pv'), put_complete=True)
    run = Cpt(EpicsSignal, 'FlyScan-MvReq-Cmd.PROC')
    abort = Cpt(EpicsSignal, 'FlyScan-Mtr.STOP')
    moving = Cpt(EpicsSignalRO, 'FlyScan-Mtr.MOVN')
    scan_in_progress = Cpt(EpicsSignalRO, 'FlyScan-Running-Sts')

    energy_lut = Cpt(EpicsSignal, write_pv='FlyLUT-Energy-SP', read_pv='FlyLUT-Energy-RB', add_prefix=('read_pv', 'write_pv'), put_complete=True)
    gap_lut = Cpt(EpicsSignal, write_pv='FlyLUT-Gap-SP', read_pv='FlyLUT-Gap-RB', add_prefix=('read_pv', 'write_pv'), put_complete=True)
    # After the LUT are updated, the calc_spline should be called.
    calc_spline = Cpt(EpicsSignal, 'CalculateSpline.PROC')
    spline_status = Cpt(EpicsSignalRO, 'FlySplineOK-RB')


class FlyScanParameters(Device):
    harmonic = Cpt(EpicsSignal, write_pv='SR:C5-ID:G1{IVU21:1}FlyHarmonic-SP', read_pv='SR:C5-ID:G1{IVU21:1}FlyHarmonic-RB', add_prefix=(), put_complete=True)

    speed = Cpt(EpicsSignal, write_pv='-Speed-SP', read_pv='-Speed-RB', add_prefix=('read_pv', 'write_pv'), put_complete=True)
    first_trigger = Cpt(EpicsSignal, write_pv='First-SP', read_pv='First-RB', add_prefix=('read_pv', 'write_pv'), put_complete=True)
    last_trigger = Cpt(EpicsSignal,  write_pv='Last-SP', read_pv='Last-RB', add_prefix=('read_pv', 'write_pv'), put_complete=True)
    num_triggers = Cpt(EpicsSignal, write_pv='NTriggers-SP', read_pv='NTriggers-RB', add_prefix=('read_pv', 'write_pv'), put_complete=True)
    trigger_width = Cpt(EpicsSignal, write_pv='TriggerWidth-SP', read_pv='TriggerWidth-RB', add_prefix=('read_pv', 'write_pv'), put_complete=True)
    id_energy_offset = Cpt(EpicsSignal, write_pv='IDOffset_eV-SP', read_pv='IDOffset_eV-RB', add_prefix=('read_pv', 'write_pv'), put_complete=True)
    timing_offset = Cpt(EpicsSignal, write_pv='TriggerOffset-SP', read_pv='TriggerOffset-RB', add_prefix=('read_pv', 'write_pv'), put_complete=True)
    trigger_count = Cpt(EpicsSignalRO, 'TriggerCount-RB')
    trigger_count_reset = Cpt(EpicsSignal, 'TriggerCount-Reset.PROC')
    num_scans = Cpt(EpicsSignal, write_pv='NScans-SP', read_pv='NScans-RB', add_prefix=('read_pv', 'write_pv'), put_complete=True)
    current_scan = Cpt(EpicsSignalRO, 'IScan-RB')
    current_scan_reset = Cpt(EpicsSignal, 'IScan-Reset.PROC')
    dwell_time = Cpt(EpicsSignal, write_pv='DwellTime-SP', read_pv='DwellTime-RB', add_prefix=('read_pv', 'write_pv'), put_complete=True)
    paused_timeout = Cpt(EpicsSignal, write_pv='PausedTimeout-SP', read_pv='PausedTimeout-RB', add_prefix=('read_pv', 'write_pv'), put_complete=True)
    scan_paused = Cpt(EpicsSignal, write_pv='Paused-SP', read_pv='Paused-RB', add_prefix=('read_pv', 'write_pv'), put_complete=True)


class HDCMParameters(Device):
    # 'ang' is for Angstrom, not angle
    ang_over_ev = Cpt(EpicsSignal, write_pv='AngOverEv-SP', read_pv='AngOverEv-RB', add_prefix=('read_pv', 'write_pv'), put_complete=True)
    d111 = Cpt(EpicsSignal, write_pv='d111-SP', read_pv='d111-SP', add_prefix=('read_pv', 'write_pv'), put_complete=True)
    delta_bragg = Cpt(EpicsSignal, write_pv='DeltaBragg-SP', read_pv='DeltaBragg-SP', add_prefix=('read_pv', 'write_pv'), put_complete=True)
    c2x_cal = Cpt(EpicsSignal, write_pv='C2XCal-SP', read_pv='C2XCal-RB', add_prefix=('read_pv', 'write_pv'), put_complete=True)
    t2_cal = Cpt(EpicsSignal, write_pv='T2Cal-SP', read_pv='T2Cal-RB', add_prefix=('read_pv', 'write_pv'), put_complete=True)
    x_offset = Cpt(EpicsSignal, write_pv='XOffset-SP', read_pv='XOffset-RB', add_prefix=('read_pv', 'write_pv'), put_complete=True)



class IDFlyDevice(Device):
    # Fly scan control
    control = Cpt(FlyScanControl, '')

    parameters = Cpt(FlyScanParameters, 'EScan')

    hdcm_parameters = Cpt(HDCMParameters, 'Fly_')

    # Fly scan parameters
    energy_motor = Cpt(EpicsMotor, 'FlyScan-Mtr')
    id_energy = Cpt(EpicsSignal, 'FlyEnergyID-RB')


try:
    id_fly_device = IDFlyDevice('SR:C5-ID:G1{IVU21:1}', name='id_fly_device')
    id_fly_device.hdcm_parameters.d111.put(energy._d_111)
    id_fly_device.hdcm_parameters.delta_bragg.put(energy._delta_bragg)
    id_fly_device.hdcm_parameters.c2x_cal.put(energy._c2xcal)
    id_fly_device.hdcm_parameters.t2_cal.put(energy._t2cal)
    id_fly_device.hdcm_parameters.x_offset.put(energy._xoffset)
except ReadOnlyError as e:
    print('Connecting to ID flyer...')
    print('  Read only error connecting to flying ID PVs!')
    print('  Continuing...')
except Exception as e:
    print(e)
    raise(e)
