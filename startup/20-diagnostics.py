print(f'Loading {__file__}...')


from ophyd import EpicsSignal, EpicsSignalRO, Device, TetrAMM, Kind
from ophyd import Component as Cpt
from ophyd.status import StatusBase, wait
from ophyd.quadem import NSLS_EM, QuadEM, QuadEMPort


# BPM1 Statistics
class BPMStats(Device):
    tot1 = Cpt(EpicsSignal, 'Stats1:Total_RBV')
    tot2 = Cpt(EpicsSignal, 'Stats2:Total_RBV')
    tot3 = Cpt(EpicsSignal, 'Stats3:Total_RBV')
    tot4 = Cpt(EpicsSignal, 'Stats4:Total_RBV')


bpm1_stats = BPMStats('XF:05IDA-BI:1{BPM:1-Cam:1}', name='bpm1_stats')


# BPM Diode
class BPMDiode(Device):
    "Beam Position Monitor Diode"
    diode0 = Cpt(EpicsSignalRO, '_Ch1')
    diode1 = Cpt(EpicsSignalRO, '_Ch2')
    diode2 = Cpt(EpicsSignalRO, '_Ch3')
    diode3 = Cpt(EpicsSignalRO, '_Ch4')
    # femto = EpicsSignal('XF:05IDA-BI:1{IM:1}Int-I')

    def trigger(self):
        # There is nothing to do. Just report that we are done.
        # Note: This really should not necessary to do --
        # future changes to PVPositioner may obviate this code.
        status = StatusBase()
        status._finished()
        return status

# bpm1 = BPMDiode('xf05bpm03:DataRead', name='bpm1')
# bpm2 = BPMDiode('xf05bpm04:DataRead', name='bpm2')
# BPM IOC disabled 2019-04-15
# bpm1 = TetrAMM('XF:05IDA-BI{BPM:3}',name='bpm1')
# bpm2 = TetrAMM('XF:05IDA-BI{BPM:4}',name='bpm2')


# Diamond BPM
class DiamondBPM(Device):
    diode_top = Cpt(EpicsSignalRO, 'Current1:MeanValue_RBV')
    diode_inb = Cpt(EpicsSignalRO, 'Current2:MeanValue_RBV')
    diode_out = Cpt(EpicsSignalRO, 'Current3:MeanValue_RBV')
    diode_bot = Cpt(EpicsSignalRO, 'Current4:MeanValue_RBV')
    sigma_top = Cpt(EpicsSignalRO, 'Current1:Sigma_RBV')
    sigma_inb = Cpt(EpicsSignalRO, 'Current2:Sigma_RBV')
    sigma_out = Cpt(EpicsSignalRO, 'Current3:Sigma_RBV')
    sigma_bot = Cpt(EpicsSignalRO, 'Current4:Sigma_RBV')
    x_pos = Cpt(EpicsSignalRO, 'PosX:MeanValue_RBV')
    y_pos = Cpt(EpicsSignalRO, 'PosY:MeanValue_RBV')
    x_sigma = Cpt(EpicsSignalRO, 'PosX:Sigma_RBV')
    y_sigma = Cpt(EpicsSignalRO, 'PosY:Sigma_RBV')


dbpm = DiamondBPM('XF:05ID-BI:1{BPM:01}:', name='dbpm')


# Setup Slit Drain Current
class SlitDrainCurrent(Device):
    t = Cpt(EpicsSignalRO, 'Current1:MeanValue_RBV')
    b = Cpt(EpicsSignalRO, 'Current2:MeanValue_RBV')
    i = Cpt(EpicsSignalRO, 'Current3:MeanValue_RBV')
    o = Cpt(EpicsSignalRO, 'Current4:MeanValue_RBV')

    def trigger(self):
        # There is nothing to do. Just report that we are done.
        # Note: This really should not necessary to do --
        # future changes to PVPositioner may obviate this code.
        status = StatusBase()
        status._finished()
        return status


wbs = SlitDrainCurrent('XF:05IDA-BI{BPM:01}AH501:', name='wbs')
pbs = SlitDrainCurrent('XF:05IDA-BI{BPM:02}AH501:', name='pbs')
ssa = SlitDrainCurrent('XF:05IDA-BI{BPM:05}AH501:', name='ssa')


# TetrAMM BPM
class BPM_TetrAMM(Device):
    "Beam Position Monitor Foil"
    channel1 = Cpt(EpicsSignalRO, 'Current1:MeanValue_RBV', kind=Kind.omitted)
    channel2 = Cpt(EpicsSignalRO, 'Current2:MeanValue_RBV', kind=Kind.omitted)
    channel3 = Cpt(EpicsSignalRO, 'Current3:MeanValue_RBV', kind=Kind.omitted)
    channel4 = Cpt(EpicsSignalRO, 'Current4:MeanValue_RBV', kind=Kind.omitted)

    x = Cpt(EpicsSignalRO, 'PosX:MeanValue_RBV')
    y = Cpt(EpicsSignalRO, 'PosY:MeanValue_RBV')
    total_current = Cpt(EpicsSignalRO, 'SumAll:MeanValue_RBV')


bpm3 = BPM_TetrAMM('XF:05IDA-BI{BPM:3}', name='bpm3')
bpm4 = BPM_TetrAMM('XF:05IDA-BI{BPM:4}', name='bpm4')


class SRX_AH501(QuadEM):
    conf = Cpt(QuadEMPort, port_name="AH501", kind="omitted")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Simplify what is read
        for d in self.read_attrs:
            getattr(self, d).kind = "omitted"
        self.read_attrs = [f'current{i}.mean_value' for i in range(1, 5)]
        for d in self.read_attrs:
            getattr(self, d).kind = Kind.normal

# bpm1 = SRX_AH501('XF:05IDA-BI{BPM:01}AH501:', name='bpm1')
# bpm2 = SRX_AH501('XF:05IDA-BI{BPM:02}AH501:', name='bpm2')
bpm5 = SRX_AH501('XF:05IDA-BI{BPM:05}AH501:', name='bpm5')


# Diamond XBPM in D-hutch
class SRX_NSLS_EM(NSLS_EM):
    sumX = Cpt(EpicsSignalRO, 'SumX:MeanValue_RBV')
    sumY = Cpt(EpicsSignalRO, 'SumY:MeanValue_RBV')
    diffX = Cpt(EpicsSignalRO, 'DiffX:MeanValue_RBV')
    diffY = Cpt(EpicsSignalRO, 'DiffY:MeanValue_RBV')
    posX = Cpt(EpicsSignalRO, 'PosX:MeanValue_RBV')
    posY = Cpt(EpicsSignalRO, 'PosY:MeanValue_RBV')

    motorX = Cpt(EpicsMotor, 'XF:05IDD-ES:1{Stg:Xbpm-Ax:X}Mtr', add_prefix=(), name='motorX')
    motorY = Cpt(EpicsMotor, 'XF:05IDD-ES:1{Stg:Xbpm-Ax:Y}Mtr', add_prefix=(), name='motorY')

    def _balanceX(self):
        return (np.abs(self.sumX.get()) - np.abs(self.diffX.get())) / np.abs(self.sumX.get())

    def _balanceY(self):
        return (np.abs(self.sumY.get()) - np.abs(self.diffY.get())) / np.abs(self.sumY.get())

    @property
    def balanceX(self, name='xbpm2_balanceX'):
       return self._balanceX()

    @property
    def balanceY(self):
       return self._balanceY()

    @property
    def balance(self):
       return np.sqrt(np.power(self.balanceX(), 2) + np.power(self.balanceY(), 2)) / np.sqrt(2)

class HACK_SRX_NSLS_EM(Device):
    current1 = Cpt(EpicsSignalRO, 'Current1:MeanValue_RBV')
    current2 = Cpt(EpicsSignalRO, 'Current2:MeanValue_RBV')
    current3 = Cpt(EpicsSignalRO, 'Current3:MeanValue_RBV')
    current4 = Cpt(EpicsSignalRO, 'Current4:MeanValue_RBV')

    sumX = Cpt(EpicsSignalRO, 'SumX:MeanValue_RBV')
    sumY = Cpt(EpicsSignalRO, 'SumY:MeanValue_RBV')
    sumT = Cpt(EpicsSignalRO, 'SumAll:MeanValue_RBV')
    diffX = Cpt(EpicsSignalRO, 'DiffX:MeanValue_RBV')
    diffY = Cpt(EpicsSignalRO, 'DiffY:MeanValue_RBV')
    posX = Cpt(EpicsSignalRO, 'PosX:MeanValue_RBV')
    posY = Cpt(EpicsSignalRO, 'PosY:MeanValue_RBV')

    bias = Cpt(EpicsSignal, 'DAC3')

    xmotor = Cpt(EpicsMotor, 'XF:05IDD-ES:1{Stg:Xbpm-Ax:X}Mtr', add_prefix=(), name='xmotor')
    ymotor = Cpt(EpicsMotor, 'XF:05IDD-ES:1{Stg:Xbpm-Ax:Y}Mtr', add_prefix=(), name='ymotor')

    def _balanceX(self):
        return (np.abs(self.sumX.get()) - np.abs(self.diffX.get())) / np.abs(self.sumX.get())

    def _balanceY(self):
        return (np.abs(self.sumY.get()) - np.abs(self.diffY.get())) / np.abs(self.sumY.get())

    @property
    def balanceX(self, name='xbpm2_balanceX'):
       return self._balanceX()

    @property
    def balanceY(self):
       return self._balanceY()

    @property
    def balance(self):
       return np.sqrt(np.power(self.balanceX(), 2) + np.power(self.balanceY(), 2)) / np.sqrt(2)

xbpm1 = HACK_SRX_NSLS_EM('XF:05ID-BI{EM:BPM1}', name='xbpm1')
xbpm2 = HACK_SRX_NSLS_EM('XF:05ID-BI{EM:BPM2}', name='xbpm2')

# EJM addition
class ScalerPreAmp(Device):
    _DEFAULT_TIMEOUT = 2

    sens_num = Cpt(EpicsSignal, 'sens_num', string=True, timeout=_DEFAULT_TIMEOUT) # XF:05IDD-CT{SR570:N}sens_num
    sens_unit = Cpt(EpicsSignal, 'sens_unit', string=True, timeout=_DEFAULT_TIMEOUT) # XF:05IDD-CT{SR570:N}sens_unit
    offset_on = Cpt(EpicsSignal, 'offset_on', string=True, timeout=_DEFAULT_TIMEOUT) # XF:05IDD-CT{SR570:N}offset_on
    offset_sign = Cpt(EpicsSignal, 'offset_sign', string=True, timeout=_DEFAULT_TIMEOUT) # XF:05IDD-CT{SR570:N}offset_sign
    offset_num = Cpt(EpicsSignal, 'offset_num', string=True, timeout=_DEFAULT_TIMEOUT) # XF:05IDD-CT{SR570:N}offset_num
    offset_unit = Cpt(EpicsSignal, 'offset_unit', string=True, timeout=_DEFAULT_TIMEOUT) # XF:05IDD-CT{SR570:N}offset_unit
    # invert = Cpt(EpicsSignal, 'invert_on', string=True, timeout=_DEFAULT_TIMEOUT)  # XF:05IDD-CT{SR570:N}invert_on
    off_u_put = Cpt(EpicsSignal, 'off_u_put', kind=Kind.omitted) # XF:05IDD-CT{SR570:N}off_u_put
    offset_u_tweak = Cpt(EpicsSignal, 'offset_u_tweak', kind=Kind.omitted) # XF:05IDD-CT{SR570:N}offset_u_tweak
    offset_cal = Cpt(EpicsSignal, 'offset_cal', kind=Kind.omitted) # XF:05IDD-CT{SR570:N}offset_cal


i0_preamp = ScalerPreAmp('XF:05IDD-CT{SR570:1}', name='i0_preamp')
im_preamp = ScalerPreAmp('XF:05IDD-CT{SR570:2}', name='im_preamp')
it_preamp = ScalerPreAmp('XF:05IDD-CT{SR570:3}', name='it_preamp')
