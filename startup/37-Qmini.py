# # BPM1 Statistics
# class BPMStats(Device):
#     tot1 = Cpt(EpicsSignal, 'Stats1:Total_RBV')
#     tot2 = Cpt(EpicsSignal, 'Stats2:Total_RBV')
#     tot3 = Cpt(EpicsSignal, 'Stats3:Total_RBV')
#     tot4 = Cpt(EpicsSignal, 'Stats4:Total_RBV')
# 
# 
# bpm1_stats = BPMStats('XF:05IDA-BI:1{BPM:1-Cam:1}', name='bpm1_stats')

class Qmini(Device):
    dwell = Cpt(EpicsSignal, ":dwell")
    num_acquire = Cpt(EpicsSignal, ":num_acquire")
    num_acquired = Cpt(EpicsSignalRO, ":num_acquired")
    SUM = Cpt(EpicsSignal, ":SUM")
    acquire = Cpt(EpicsSignal, ":acquire")
    spectrum = Cpt(EpicsSignalRO, ":spectrum")
    wavelength = Cpt(EpicsSignalRO, ":wavelength")

    path = Cpt(EpicsSignal, "-Saver:root")
    filename = Cpt(EpicsSignal, "-Saver:filename")
    capture = Cpt(EpicsSignal, "-Saver:capture")

    def trigger(self):
        # There is nothing to do. Just report that we are done.
        # Note: This really should not necessary to do --
        # future changes to PVPositioner may obviate this code.
        st = yield from mov(self.num_acquire, 1)
        print(f"{st=}")
        print(f"{st.__dir__()=}")
        st[0].wait(5)
        # st = yield from mov(self.acquire, 1)
        # yield from bps.sleep(self.dwell.get())
        # st[0].wait(self.dwell.get()*3)
        # print(f"{st=}")
        # print(f"{st.__dir__()=}")
        # return st

        def check_value(*, old_value, value, **kwargs):
            return (old_value == 1 and value == 0)

        status = SubscriptionStatus(self.acquire, check_value)
        self.acquire.set(1)
        # status = DeviceStatus(self)
        # status.set_finished()
        return status


    # def trigger_and_read(self):
    #     # There is nothing to do. Just report that we are done.
    #     # Note: This really should not necessary to do --
    #     # future changes to PVPositioner may obviate this code.
    #     st = yield from mov(self.num_acquire, 1)
    #     print(f"{st=}")
    #     st[0].wait(5)
    #     st = yield from mov(self.acquire.get(), 1)
    #     yield from bps.sleep(self.dwell.get())
    #     st[0].wait(self.dwell*3)

    #     status = StatusBase()
    #     status._finished()
    #     return status


qmini = Qmini("XF:05IDD-Qmini", name="qmini")

def qmini_per_step(detectors, motor, step):
    def move():
        grp = _short_uid('set')
        yield Msg('checkpoint')
        yield Msg('set', motor, step, group=grp)
        yield Msg('wait', None, group=grp)

    yield from move()
    # Open and close the fast shutter (Mo Foil) between XANES points
    # Open the shutter
    # yield from mv(shut_d, 0)
    # yield from mv(shut_d.request_open, 1)
    yield from bps.sleep(1)
    # yield from bps.sleep(1.0)
    # Step? trigger xspress3
    yield from trigger_and_read(list(detectors) + [motor])
    # Close the shutter
    # yield from mv(shut_d, 1)
    # yield from mv(shut_d.request_open, 0)

