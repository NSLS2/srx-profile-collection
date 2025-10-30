print(f'Loading {__file__}...')

from bluesky.suspenders import (SuspendFloor, SuspendCeil,
                                SuspendBoolHigh, SuspendBoolLow)
import bluesky.plans as bp
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp

from ophyd.sim import FakeEpicsSignal


class SRXSuspCryo(SuspendBoolHigh):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tripped_message= \
            "\n\n----\nCryocooler EPS activated! Contact beamline staff!\n----\n"

    def __repr__(self):
        return self._sig.name

    def _should_resume(self, value):
        self.RE.abort()
        return True 


def shuttergenerator(shutter, value):
    return (yield from bpp.rewindable_wrapper(bps.mv(shutter, value), False))

def another_generator(pv, value):
    return (yield from bps.mov(pv, value))


# Ring current suspender
susp_rc = SuspendFloor(ring_current, 200, resume_thresh=400, sleep=10*60,
                       pre_plan=list(shuttergenerator(shut_b, 'Close')),
                       post_plan=list(shuttergenerator(shut_b, 'Open')))

# Cryo cooler suspender
susp_cryo = SuspendCeil(cryo_v19, 0.8, resume_thresh=0.2, sleep=15*60,
                        pre_plan=list(shuttergenerator(shut_b, 'Close')),
                        post_plan=list(shuttergenerator(shut_b, 'Open')))
susp_cryo_eps = SRXSuspCryo(EpicsSignalRO("XF:05IDA-UT{Cryo:1}:ALARMSTATUS",
                                          name="Cryocooler EPS"))

# Shutter status suspender
susp_shut_fe = SuspendBoolHigh(EpicsSignalRO(shut_fe.status.pvname, name="FE shutter"),
                               sleep=10)
susp_shut_a  = SuspendBoolHigh(EpicsSignalRO(shut_a.status.pvname, name="A shutter"),
                               sleep=10)
susp_shut_b  = SuspendBoolHigh(EpicsSignalRO(shut_b.status.pvname, name="B shutter"),
                               sleep=10)

# HDCM bragg temperature suspender
susp_dcm_bragg_temp = SuspendCeil(dcm.temp_pitch, 120, resume_thresh=118, sleep=1)


# Install suspenders
RE.install_suspender(susp_rc)
# RE.install_suspender(susp_shut_b)
RE.install_suspender(susp_dcm_bragg_temp)
RE.install_suspender(susp_cryo_eps)
