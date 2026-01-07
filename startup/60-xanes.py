print(f'Loading {__file__}...')

import itertools
import collections
from collections import deque

import numpy as np
import time as ttime
import matplotlib.pyplot as plt

import bluesky.plans as bp
from bluesky.plans import list_scan
import bluesky.plan_stubs as bps
from bluesky.plan_stubs import mv
from bluesky.preprocessors import (finalize_wrapper, subs_wrapper)
from bluesky.utils import short_uid as _short_uid
from scipy.interpolate import make_interp_spline

# From flying-undulator branch
from numpy.lib.stride_tricks import as_strided
from ophyd.status import SubscriptionStatus
from ophyd.sim import NullStatus

# Kafka can throw a warning if the document is too large
# Adding this is an attempt to prevent the document (really long) being output to screen
# warnings.filterwarnings(action="ignore", message="MSG_SIZE_TOO_LARGE")


@append_srx_kwargs_md
@parameter_annotation_decorator({
    "parameters": {
        "det_xs": {"default": "'xs'"},
    }
})
def xanes_plan(erange=[], estep=[], acqtime=1.,
               md=None,
               det_xs=xs,
               harmonic=1,
               detune=0,
               align=False,
               align_at=None,
               roinum=1,
               shutter=True,
               per_step=None,
               reverse=False,
               vlm_snapshot=True):
    '''
    erange (list of floats): energy ranges for XANES in eV, e.g. erange = [7112-50, 7112-20, 7112+50, 7112+120]
    estep  (list of floats): energy step size for each energy range in eV, e.g. estep = [2, 1, 5]
    acqtime (float): acqusition time to be set for both xspress3 and preamplifier
    samplename (string): sample name to be saved in the scan metadata

    det_xs (xs3 detector): the xs3 detector used for the measurement
    harmonic (odd integer): when set to 1, use the highest harmonic achievable automatically.
                                    when set to an odd integer, force the XANES scan to use that harmonic
    detune:  add this value to the gap of the undulator to reduce flux [keV]
    align:  perform peakup_fine before scan [bool]
    align_at:  energy at which to align, default is average of the first and last energy points
    roinum: select the roi to be used to calculate the XANES spectrum
    shutter:  instruct the scan to control the B shutter [bool]
    per_step:  use a custom function for each energy point
    '''
    # Make sure user provided correct input
    if (erange is []):
        raise AttributeError("An energy range must be provided in a list by means of the 'erange' keyword.")
    if (estep is []):
        raise AttributeError("A list of energy steps must be provided by means of the 'esteps' keyword.")
    if (not isinstance(erange,list)) or (not isinstance(estep,list)):
        raise TypeError("The keywords 'estep' and 'erange' must be lists.")
    if (len(erange)-len(estep) != 1):
        raise ValueError("The 'erange' and 'estep' lists are inconsistent; " \
                         + 'c.f., erange = [7000, 7100, 7150, 7500], estep = [2, 0.5, 5] ')
    if (type(roinum) is not list):
        roinum = [roinum]
    if (detune != 0):
        yield from abs_set(energy.detune, detune)

    # Convert erange and estep to numpy array
    ept = np.array([])
    erange = np.array(erange)
    estep = np.array(estep)
    # Calculation for the energy points
    for i in range(len(estep)):
        ept = np.append(ept, np.arange(erange[i], erange[i+1], estep[i]))
    ept = np.append(ept, np.array(erange[-1]))
    if reverse:
        ept = ept[::-1]

    # Debugging, is this needed? is this recorded in scanoutput?
    # Convert energy to bragg angle
    egap = np.array(())
    ebragg = np.array(())
    exgap = np.array(())
    for i in ept:
        eg, eb, ex = energy.forward(i)
        egap = np.append(egap, eg)
        ebragg = np.append(ebragg, eb)
        exgap = np.append(exgap, ex)

    # Register the detectors
    det_xs.mode = SRXMode.step
    # det = [ring_current, sclr1, xbpm2, det_xs] # EJM removed xpbm2 when disconnected. June 30th 2025
    det = [ring_current, sclr1, det_xs]
    # Setup xspress3
    yield from abs_set(det_xs.external_trig, False)
    yield from abs_set(get_me_the_cam(det_xs).acquire_time, acqtime)
    yield from abs_set(det_xs.total_points, len(ept))

    # Record relevant meta data in the Start document, defined in 90-usersetup.py
    # Add user meta data
    md = get_stock_md(md)
    md['scan']['type'] = 'XAS_STEP'
    md['scan']['ROI'] = roinum
    md['scan']['dwell'] = acqtime
    md['scan']['scan_input'] = str(np.around(erange, 2)) + ', ' + str(np.around(estep, 2))
    md_dets = list(det)
    if vlm_snapshot:
        md_dets = md_dets + [nano_vlm]
    get_det_md(md, md_dets)
    # Fix for the exporter
    md['detectors'] = [det.name for det in md_dets]

    # Setup the scaler
    # TODO: Is there a better way to consider this?
    # Right now, this will clear 'present_time' in stage_sigs
    # This will get the desired result but is there a time when
    # using stage_sigs is correct/better
    sclr1.stage_sigs.pop('preset_time', None)
    yield from abs_set(sclr1.preset_time, acqtime)

    # Setup DCM/energy options
    if (harmonic != 1):
        yield from abs_set(energy.harmonic, harmonic)

    # Prepare to peak up DCM at middle scan point
    if (align_at is not None):
        align = True
    if (align is True):
        if (align_at is None):
            align_at = 0.5*(ept[0] + ept[-1])
            print("Aligning at ", align_at)
            yield from abs_set(energy, align_at, wait=True)
        else:
            print("Aligning at ", align_at)
            yield from abs_set(energy, float(align_at), wait=True)

    # Peak up DCM at first scan point
    if (align is True):
        yield from peakup(shutter=shutter)

    # Setup the live callbacks
    livecallbacks = []
    # Setup Raw data
    livetableitem = ['energy_energy', 'sclr_i0', 'sclr_it']
    roi_name = 'roi{:02}'.format(roinum[0])

    if hasattr(det_xs, 'cam'):
        roi_key = [
            det_xs_channel.get_mcaroi(mcaroi_number=roinum[0]).total_rbv.name
            for det_xs_channel
            in det_xs.iterate_channels()
        ]
    else:
        roi_key = []
        roi_key.append(getattr(det_xs.channel1.rois, roi_name).value.name)
        try:
            roi_key.append(getattr(det_xs.channel2.rois, roi_name).value.name)
            roi_key.append(getattr(det_xs.channel3.rois, roi_name).value.name)
            roi_key.append(getattr(det_xs.channel4.rois, roi_name).value.name)
        except NameError:
            pass


    def time_per_point(name, doc, st=ttime.time()):
        ## Don't do this. Make a proper fix.
        try:
            if (doc[0] == "event_page"):
                if ('seq_num' in doc.keys()):
                    scanrecord.time_remaining.put((doc['time'] - st) / doc['seq_num'] *
                                              (len(ept) - doc['seq_num']) / 3600)
        except Exception:
            pass

    livetableitem.append(roi_key[0])
    livecallbacks.append(LiveTable(livetableitem))
    livecallbacks.append(time_per_point)
    liveploty = roi_key[0]
    liveplotx = energy.energy.name

    def my_factory(name):
        fig = plt.figure(num=name)
        ax = fig.gca()
        return fig, ax

    # liveplotfig = plt.figure('Raw XANES')
    # livecallbacks.append(LivePlot(liveploty, x=liveplotx, fig=liveplotfig))
    livecallbacks.append(HackLivePlot(liveploty, x=liveplotx,
                                      fig_factory=partial(my_factory, name='Raw XANES')))

    # Setup normalization
    liveploty = 'sclr_i0'
    i0 = 'sclr_i0'
    # liveplotfig2 = plt.figure('I0')
    # livecallbacks.append(LivePlot(liveploty, x=liveplotx, fig=liveplotfig2))
    livecallbacks.append(HackLivePlot(liveploty, x=liveplotx,
                                      fig_factory=partial(my_factory, name='I0')))

    # Setup normalized XANES
    # livenormfig = plt.figure('Normalized XANES')
    # livecallbacks.append(NormalizeLivePlot(roi_key[0], x=liveplotx, norm_key = i0, fig=livenormfig))
    livecallbacks.append(NormalizeLivePlot(roi_key[0], x=liveplotx, norm_key = i0,
                                           fig_factory=partial(my_factory, name='Normalized XANES')))


    def at_scan(name, doc):
        scanrecord.current_scan.put(doc['uid'][:6])
        scanrecord.current_scan_id.put(str(doc['scan_id']))
        scanrecord.current_type.put(md['scan']['type'])
        scanrecord.scanning.put(True)


    def finalize_scan():
        yield from abs_set(energy.move_c2_x, True)
        yield from abs_set(energy.harmonic, 1)
        # if (shutter is True):
        #     yield from mv(shut_b,'Close')
        yield from check_shutters(shutter, 'Close')
        if (detune is not None):
            yield from abs_set(energy.detune, 0)
        scanrecord.scanning.put(False)


    energy.move(ept[0])

    # Adding vlm options to modified list scan
    @run_decorator(md=md)
    @vlm_decorator(vlm_snapshot, after=True)
    def myscan():
        yield from mod_list_scan(det, energy, list(ept), per_step=per_step, run_agnostic=True)

    # myscan = mod_list_scan(det, energy, list(ept), per_step=per_step, md=md)
    # myscan must be called to return the generators
    myscan = finalize_wrapper(myscan(), finalize_scan)

    # Open B shutter
    yield from check_shutters(shutter, 'Open')

    return (yield from subs_wrapper(myscan, {'all' : livecallbacks,
                                               'start' : at_scan}))

# Alias
xas_step = xanes_plan


def xanes_batch_plan(xypos=[], erange=[], estep=[], acqtime=1.0,
                     repeat_point=1, waittime=10, peakup_N=2, peakup_E=None):
    """
    Setup a batch XANES scan at multiple points.
    This scan can also run peakup() between points.

    xypos       <list>  A list of points to run XANES scans
                        Each point can be listed as 2 coordinates [x, y]
                        or 3 coordinates [x, y, z]
                        e.g. xypos=[[0, 0, 0], [5, -5], [45, 45, -10]]
    erange      <list>  A list of energy points to send to the XANES plan
                        similar to a typical xanes_plan
    estep       <list>  A list of energy steps to send to the XANES plan
                        similar to a typical xanes_plan
    acqtime     <float> Acquisition time for each data point.
    repeat_point<int>   Repeat a particular point N times
    peakup_N    <int>   Run a peakup every peakup_N scans. Default is no peakup
    peakup_E    <float> The energy to run peakup at. Default is current energy

    """

    # Check positions
    if (xypos == []):
        print('You need to enter positions.')
        return

    # Check erange and estep
    if (erange == []):
        print('You need to enter erange.')
        return
    if (estep == []):
        print('You need to enter estep.')
        return

    # Get current energy and use it for peakup
    if (peakup_E == None):
        peakup_E = energy.position.energy

    # Convert keV to eV
    if (peakup_E < 1000):
        peakup_E = peakup_E * 1000

    # Loop through positions
    N = len(xypos)
    for i in range(N):
        print(f'Moving to:')
        print(f'\tx = {xypos[i][0]}')
        print(f'\ty = {xypos[i][1]}')
        yield from mov(nano_stage.sx, xypos[i][0])
        yield from mov(nano_stage.sy, xypos[i][1])
        if (len(xypos[i]) == 3):
            print(f'\tz = {xypos[i][2]}')
            yield from mov(nano_stage.sz, xypos[i][2])

        # Move above edge and peak up
        if (i % peakup_N == 0 and i != 0):
            yield from mov(energy, peakup_E)
            yield from peakup()

        # Run the energy scan
        for j in range(repeat_point):
            print(f"Scan {i+1}/{repeat_point}...")
            yield from xanes_plan(erange=erange, estep=estep, acqtime=acqtime)

        # Wait
        if (i != (N-1)):
            print(f'Scan complete. Waiting {waittime} seconds...')
            yield from bps.sleep(waittime)


def hfxanes_ioc(erange=[], estep=[], acqtime=1.0, samplename='', filename='',
                harmonic=1, detune=0, align=False, align_at=None,
                roinum=1, shutter=True, per_step=None, waittime=0):
    '''
    invokes hf2dxrf repeatedly with parameters provided separately.
        waittime                [sec]       time to wait between scans
        shutter                 [bool]      scan controls shutter
        align                   [bool]      optimize beam location on each scan
        roinum                  [1,2,3]     ROI number for data output

    '''

    scanlist = [ scanrecord.scan15, scanrecord.scan14, scanrecord.scan13,
                 scanrecord.scan12, scanrecord.scan11, scanrecord.scan10,
                 scanrecord.scan9, scanrecord.scan8, scanrecord.scan7,
                 scanrecord.scan6, scanrecord.scan5, scanrecord.scan4,
                 scanrecord.scan3, scanrecord.scan2, scanrecord.scan1,
                 scanrecord.scan0 ]
    Nscan = 0
    for scannum in range(len(scanlist)):
        thisscan = scanlist.pop()
        Nscan = Nscan + 1
        if (thisscan.Eena.get() == 1):
            scanrecord.current_scan.put('Scan {}'.format(Nscan))
            erange = [thisscan.e1s.get(), thisscan.e2s.get(), thisscan.e3s.get(), thisscan.efs.get()]
            estep = [thisscan.e1i.get(), thisscan.e2i.get(), thisscan.e3i.get()]
            waittime = thisscan.Ewait.get()

            xstart = thisscan.p1s.get()
            ystart = thisscan.p2s.get()
            # Move stages to the next point
            yield from mv(hf_stage.x, xstart,
                          hf_stage.y, ystart)
            acqtime = thisscan.acq.get()

            hfxanes_gen = yield from xanes_plan(erange=erange, estep=estep, acqtime=thisscan.acq.get(),
                                                samplename=thisscan.sampname.get(), filename=thisscan.filename.get(),
                                                roinum=int(thisscan.roi.get()), detune=thisscan.detune.get(), shutter=shutter)
            if (len(scanlist) != 0):
                yield from bps.sleep(waittime)
    scanrecord.current_scan.put('')


def fast_shutter_per_step(detectors, motor, step):
    def move():
        grp = _short_uid('set')
        yield Msg('checkpoint')
        yield Msg('set', motor, step, group=grp)
        yield Msg('wait', None, group=grp)

    yield from move()
    # Open and close the fast shutter (Mo Foil) between XANES points
    # Open the shutter
    # yield from mv(shut_d, 0)
    yield from mv(shut_d.request_open, 1)
    yield from bps.sleep(0.010)
    # yield from bps.sleep(1.0)
    # Step? trigger xspress3
    yield from trigger_and_read(list(detectors) + [motor])
    # Close the shutter
    # yield from mv(shut_d, 1)
    yield from mv(shut_d.request_open, 0)


class FlyerIDMono(Device):
    def __init__(self, flying_dev, zebra, xs_detectors, scaler, pulse_cpt=None, pulse_width=0.01, paused_timeout=120):
        """Instantiate a flyer based on ID-Mono coordinated motion.

        Parameters
        ----------
        flying_dev : IDFlyDevice
            ID-Mono fly device that has controls of the DCM and ID energies

        zebra : SRXZebra
            zebra ophyd object

        xs_detectors : list
            a list of ophyd objects for corresponding xspress3 detectors

        scaler : SRXScaler
            an ophyd object for the scaler detector

        pulse_cpt : str
            an ophyd component name corresponding to the pulse signal

        pulse_width : float
            the pulse width in seconds, used for zebra

        paused_timeout : float
            the timeout to wait between the steps until the scan is interrupted if the "unpause" button is not pressed.
        """
        super().__init__("", parent=None, name="flyer_id_mono")

        self.flying_dev = flying_dev
        self.zebra = zebra
        self.xs_detectors = xs_detectors
        
        # TODO: Check this.
        for det in self.xs_detectors:
            det.mode = SRXMode.fly
        
        self.scaler = scaler
        self._staging_delay = 0.010

        # LUTs
        self.lut_u = EpicsSignal(
                         "SR:C5-ID:G1{IVU21:1}FlyLUT-Gap-RB",
                         write_pv="SR:C5-ID:G1{IVU21:1}FlyLUT-Gap-SP",
                         name="lut_u",
                         kind="omitted"
                         )
        self.lut_e = EpicsSignal(
                         "SR:C5-ID:G1{IVU21:1}FlyLUT-Energy-RB",
                         write_pv="SR:C5-ID:G1{IVU21:1}FlyLUT-Energy-SP",
                         name="lut_e",
                         kind="omitted"
                         )

        # The pulse width has to be set both in Zebra and the Scan Engine.
        if pulse_cpt is None:
            raise RuntimeError(f'pulse_cpt cannot be None. Please provide a valid component name.')
        self.pulse_cpt = pulse_cpt
        self.pulse_width = pulse_width

        self.num_scans = None
        self.num_triggers = None

        self.paused_timeout = paused_timeout
        self._continue_after_pausing = True

        # Flyer infrastructure parameters.
        self._traj_info = {}
        self._array_size = {}
        self._datum_ids = []

    def stage(self):
        # total_points = self.num_scans * self.num_triggers
        if self.num_triggers is None:
            print(f'You must set {self.name}.num_triggers before staging!')
            raise Exception
        total_points = self.num_triggers

        for xs_det in self.xs_detectors:
            xs_det.hdf5.file_write_mode.put('Capture')
            xs_det.external_trig.put(True)
            xs_det.total_points.put(total_points)
            xs_det.spectra_per_point.put(1)
            xs_det.stage()
            xs_det.cam.acquire.put(1)
            #xs_det.mode = SRXMode.fly

        # Scaler config
        # self.scaler.count_mode.put(0)  # put SIS3820 into single count (not autocount) mode
        self.scaler.stop_all.put(1)  # stop scaler
        ttime.sleep(0.050)
        self.scaler.nuse_all.put(2*total_points)
        ttime.sleep(0.050)
        self._stage_with_delay()
        self.scaler.erase_start.put(1)  # This is basically triggering the detector so it should be done last

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
        for xs_det in self.xs_detectors:
            xs_det.cam.acquire.put(0)
            xs_det.hdf5.capture.put(0)  # this is to save the file is the number of collected frames is less than expected
            xs_det.cam.trigger_mode.put('Internal')
            xs_det.unstage()

        # print(f"{print_now()}: before unstaging scaler")
        self.scaler.stop_all.put(1)
        self.scaler.count_mode.put(1)
        self.scaler.read_attrs = ["channels.chan2", "channels.chan3", "channels.chan4"]
        # self.scaler.count_mode.put(1)  # return SIS3820 into autocount (not single count) mode
        # print(f"{print_now()}: after unstaging scaler")

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

    def kickoff(self, *args, **kwargs):

        # print('In kickoff...')
        # Reset zebra to clear the data entries.
        self.zebra.pc.block_state_reset.put(1)

        # Arm zebra.
        # self.zebra.pc.arm.put(1)  # is this necessary? we don't use PC

        # PULSE tab of the Zebra CSS:
        # getattr(self.zebra, self.pulse_cpt).input_addr.put(1)            # 'Input' in CSS, 1=IN1_TTL
        # getattr(self.zebra, self.pulse_cpt).input_edge.put(0)            # 'Trigger on' in CSS, 0=Rising, 1=Falling
        # getattr(self.zebra, self.pulse_cpt).delay.put(0.0)               # 'Delay before' in CSS
        getattr(self.zebra, self.pulse_cpt).width.put(self.pulse_width)  # 'First Pulse' in CSS
        # getattr(self.zebra, self.pulse_cpt).time_units.put('s')          # 'Time Units' in CSS

        # SYS tab of the Zebra CSS
        # for out in [1, 2, 3, 4]:
        # for out in [1, 2, 4]:
        #     getattr(self.zebra, f'output{out}').ttl.addr.put(52)          # 'OUTx TTL' in CSS

        # TESTING
        # self.zebra.output3.ttl.addr.put(36)

        # self.zebra.pulse3.input_addr.put(52)
        # self.zebra.pulse3.input_edge.put(0)
        # self.zebra.pulse3.time_units.put('ms')
        # self.zebra.pulse3.width.put(0.100)
        # self.zebra.pulse3.delay.put(0.0)

        # self.zebra.pulse4.input_addr.put(52)
        # self.zebra.pulse4.input_edge.put(1)
        # self.zebra.pulse4.time_units.put('ms')
        # self.zebra.pulse4.width.put(0.100)
        # self.zebra.pulse4.delay.put(0.0)

        width_s = self.pulse_width
        speed = self.flying_dev.parameters.speed.get()

        self.num_scans = num_scans = self.flying_dev.parameters.num_scans.get()
        self.num_triggers = num_triggers = int(self.flying_dev.parameters.num_triggers.get())

        self.flying_dev.parameters.paused_timeout.put(self.paused_timeout)

        self._traj_info.update({
            'num_triggers': num_triggers,
            'energy_start': self.flying_dev.parameters.first_trigger.get(),
            'energy_stop': self.flying_dev.parameters.last_trigger.get(),
            })

        # print('Staging...')
        self.stage()

        # Convert to eV/s.
        # width_ev = width_s * speed
        # self.flying_dev.parameters.trigger_width.put(width_ev)

        # print(f'Enabling fly scan')
        st = self.flying_dev.control.set("enable")
        try:
            st.wait(10)
        except WaitTimeoutError as ex:
            print("Timeout while requesting control of IVU!")
            print("Trying again...")

            print("  disabling...", end="", flush=True)
            st = self.flying_dev.control.set("disable")
            st.wait(10)
            print("done")

            print("  requesting...", end="", flush=True)
            st = self.flying_dev.control.set("enable")
            st.wait(10)
            print("done")
        except Exception as ex:
            raise ex


        # Reset the trigger count and current scan:
        self.flying_dev.parameters.trigger_count_reset.put(1)
        self.flying_dev.parameters.current_scan_reset.put(1)

        ttime.sleep(1.0)  # Wait 1 second because the control PV updates at 1 Hz

        # Main RUN command:
        self.flying_dev.control.run.put(1)

        self.status = self.flying_dev.control.scan_in_progress

        def callback(value, old_value, **kwargs):
            # print(f'{print_now()} in kickoff: {old_value} ---> {value}')
            if int(round(old_value)) == 0 and int(round(value)) == 1:
                return True
            return False

        status = SubscriptionStatus(self.status, callback)
        return status

    def complete(self, *args, **kwargs):
        if self.xs_detectors[0]._staged.value == 'no':

            # Note: this is a way to stop the scan on the fly.
            if self.flying_dev.parameters.num_scans.get() == 0:
                self._continue_after_pausing = False
                self.flying_dev.control.abort.put(1)

            if self._continue_after_pausing:
                self.stage()
                self.flying_dev.parameters.scan_paused.put(0)

        def _complete_detectors():
            # print(f"{print_now()} run 'complete' on detectors.")
            # ttime.sleep(0.5)
            for xs_det in self.xs_detectors:
                # print(f"{print_now()} before erase in '_complete_detectors'.")
                # xs_det.cam.erase.put(1)
                # print(f"{print_now()} after erase in '_complete_detectors'.")
                # print(f"{xs_det.name=}")
                xs_det.complete()
            # print(f"{print_now()} done with 'complete' on detectors.")

        def callback_paused(value, old_value, **kwargs):
            # print(f"{print_now()} 'callback_paused' in complete:  scan_paused: {old_value} ---> {value}")
             # 1=Paused, 0=Not Paused
            if int(round(old_value)) == 0 and int(round(value)) == 1:
                _complete_detectors()
                return True
            return False

        def callback_all_scans_done(value, old_value, **kwargs):
            # print(f"{print_now()} 'callback_all_scans_done' in complete:  current_scan: {old_value} ---> {value}")
            if value == self.flying_dev.parameters.num_scans.get():  # last scan in the series, no pausing happens
                # print("  Last scan in the series.")
                _complete_detectors()
                self.zebra.pc.disarm.put(1)
                return True
            return False

        current_scan = self.flying_dev.parameters.current_scan.get()
        num_scans = self.flying_dev.parameters.num_scans.get()

        if  current_scan + 1 < num_scans:  # last scan
            status_paused = SubscriptionStatus(self.flying_dev.parameters.scan_paused, callback_paused, run=False)
            return status_paused
        elif current_scan + 1 == num_scans:
            status_all_scans_done = SubscriptionStatus(self.flying_dev.parameters.current_scan, callback_all_scans_done, run=False)
            return status_all_scans_done
        else:
            return NullStatus()

    # TODO: Fix the configuration (also for v2).
    # def describe_configuration(self, *args, **kwargs):
    #     ret = {}
    #     for xs_det in self.xs_detectors:
    #         ret[f'{xs_det}.name'] = xs_det.describe_configuration()
    #     return ret

    def describe_collect(self, *args, **kwargs):
        # print(f"\n\n{print_now()}: describe_collect started")
        return_dict = {}
        if True:
        # for scan_num in range(self.num_scans):
            current_scan = self.flying_dev.parameters.current_scan.get()

            # print(f"{print_now()}: current_scan: {current_scan}")

            formatted_scan_num = f"scan_{current_scan:03d}"
            return_dict[formatted_scan_num] = \
                {'energy': {'source': self.flying_dev.name,
                            'dtype': 'array',
                            # We need just 1 scalar value for the energy.
                            # 'shape': [self._traj_info['num_triggers']]},
                            # TODO: double-check the shape is right for databroker v2.
                            'shape': [self.num_triggers]},
                 'i0_time': {'source': 'scaler', 'dtype': 'array', 'shape': [self.num_triggers]},
                 'i0': {'source': 'scaler', 'dtype': 'array', 'shape': [self.num_triggers]},
                 'im': {'source': 'scaler', 'dtype': 'array', 'shape': [self.num_triggers]},
                 'it': {'source': 'scaler', 'dtype': 'array', 'shape': [self.num_triggers]},
                 # f'{self.detector.name}_image': {'source': '...',
                 #           'dtype': 'array',
                 #           'shape': [self._array_size['height'],
                 #                     self._array_size['width']],
                 #           'external': 'FILESTORE:'}
                }
            
            for xs_det in self.xs_detectors:
                # TODO
                
                # for channel in xs_det.channels.keys():
                for channel in xs_det.iterate_channels():
                    return_dict[formatted_scan_num][f'{xs_det.name}_channel{channel.channel_number:02}'] = {'source': 'xspress3',
                                                                        'dtype': 'array',
                                                                        # The shape will correspond to a 1-D array of 4096 bins from xspress3.
                                                                        'shape': [
                                                                                  # in multi-pass mode this is the number of pointsin one pass
                                                                                  self.num_triggers,
                                                                                  # We don't need the total number of frames here.
                                                                                  # xs_det.settings.num_images.get(),
                                                                                  #
                                                                                  # The height corresponds to a number of channels, but we only need one here.
                                                                                  # xs_det.hdf5.array_size.height.get(),
                                                                                  #
                                                                                  xs_det.hdf5.array_size.width.get()],
                                                                        'external': 'FILESTORE:'}
        import pprint
        # pprint.pprint(return_dict)

        # print(f"\n\n{print_now()}: describe_collect ended")

        return return_dict

    def collect(self, *args, **kwargs):

        # TODO: test that feature.
        if not self._continue_after_pausing:
            return {}

        energy_start = self._traj_info['energy_start']
        energy_stop = self._traj_info['energy_stop']
        num_triggers = self._traj_info['num_triggers']

        # if len(self._datum_ids) != num_triggers:
        #     raise RuntimeError(f"The number of collected datum ids ({self._datum_ids}) "
        #                        f"does not match the number of triggers ({num_triggers})")

        ttime.sleep(self.pulse_width + 0.1)

        orig_read_attrs = self.scaler.read_attrs
        self.scaler.read_attrs = ['mca1', 'mca2', 'mca3', 'mca4']
        # print(orig_read_attrs)

        total_points = self.num_scans * self.num_triggers

        print(f"{print_now()}: before while loop in collect scaler data")
        flag_collecting_data = 0
        while (flag_collecting_data < 5):
            scaler_mca_data = self.scaler.read()
            i0_time = scaler_mca_data[f"{self.scaler.name}_mca1"]['value']
            i0 = scaler_mca_data[f"{self.scaler.name}_mca2"]['value']
            im = scaler_mca_data[f"{self.scaler.name}_mca3"]['value']
            it = scaler_mca_data[f"{self.scaler.name}_mca4"]['value']

            # print(f'{i0_time.shape[0]}\t?=\t{2*self.num_triggers}')
            if i0_time.shape[0] == 2*self.num_triggers:
                break
            flag_collecting_data += 1
            ttime.sleep(0.2)
            # print(f'({flag_collecting_data+1}/5) Waiting to collect all scaler data...')
        print(f"{print_now()}: after while loop in collect scaler data")

        self.scaler.read_attrs = orig_read_attrs
        # print(self.scaler.read_attrs)

        # print(f"Length of 'i0_time': {len(i0_time)}")
        # print(f"Length of 'i0'     : {len(i0)}")
        # print(f"Length of 'im'     : {len(im)}")
        # print(f"Length of 'it'     : {len(it)}")

        i0_time = i0_time[1::2]
        i0 = i0[1::2]
        im = im[1::2]
        it = it[1::2]

        # print(f"Truncated length of 'i0_time': {len(i0_time)}")
        # print(f"Truncated length of 'i0'     : {len(i0)}")
        # print(f"Truncated length of 'im'     : {len(im)}")
        # print(f"Truncated length of 'it'     : {len(it)}")

        if len(i0_time) != len(i0) != len(im) != len(it):
            # print(f'{len(i0_time)=}')
            raise RuntimeError(f"Lengths of the collected arrays are not equal")
        # print(f"{len(i0_time)=}")
        # print(f"{num_triggers=}")
        if len(i0_time) != num_triggers:
            # I don't understand why I can't do this in a for-loop with a list
            # for d in [i0_time, i0, im, it]:
            #     d = np.concatenate((d, np.ones((num_triggers-len(d),))))
            i0_time = np.concatenate((i0_time, np.ones((num_triggers-len(i0_time),))))
            i0 = np.concatenate((i0, np.ones((num_triggers-len(i0),))))
            im = np.concatenate((im, np.ones((num_triggers-len(im),))))
            it = np.concatenate((it, np.ones((num_triggers-len(it),))))
            # print(f'{len(i0_time)=}')

        # print(f"{print_now()}: before unstage of xs in collect")

        # Unstage xspress3 detector(s).
        print(f"{print_now()}: before unstaging xs")
        self.unstage()
        print(f"{print_now()}: after unstaging xs")

        # print(f"{print_now()}: after unstage of xs in collect")

        # Deal with the direction of energies for bi-directional scan.
        # BlueSky@SRX [27]: id_fly_device.control.scan_type.get(as_string=True)
        # Out[27]: 'Bidirectional'

        # BlueSky@SRX [28]: id_fly_device.control.scan_type.enum_strs
        # Out[28]: ('Unidirectional', 'Bidirectional')

        even_direction = np.linspace(energy_start, energy_stop, num_triggers)
        odd_direction =  even_direction[::-1]

        scan_type = self.flying_dev.control.scan_type.get(as_string=True)
        current_scan = self.flying_dev.parameters.current_scan.get()
        # print(f"{print_now()} the scan is {scan_type}; current scan: {current_scan}")

        direction = even_direction
        if scan_type == "Bidirectional":
            if (current_scan + 1) % 2 == 1:  # at this point the current scan number is already incremented
                direction = odd_direction
                # print(f"{print_now()} reversing the energy axis: {direction[0]} --> {direction[-1]}")


        # print(f"{print_now()}: In possible long loop...")
        # for ii, energy in enumerate(direction):
        #     # print(f"  {print_now()}: {energy}")
        #     for xs_det in self.xs_detectors:
        #         # print(f"  {print_now()}: {xs_det.name}")
        #         now = ttime.time()

        #         data = {
        #             'energy': energy,
        #             'i0_time': i0_time[ii],
        #             'i0': i0[ii],
        #             'im': im[ii],
        #             'it': it[ii],
        #         }
        #         timestamps = {
        #             'energy': now,
        #             'i0_time': now,
        #             'i0': now,
        #             'im': now,
        #             'it': now,
        #         }
        #         filled = {}
        #         for jj, channel in enumerate(xs_det.iterate_channels()):
        #             # print(f"  {print_now()}: {channel.name}")
        #             key = channel.name
        #             idx = jj + ii * len(xs_det.channel_numbers)
        #             timestamps[key] = now
        #             filled[key] = False
        #             try:
        #                 # print(f"{xs_det._datum_ids=}")
        #                 data[key] = xs_det._datum_ids[idx]
        #             except IndexError:
        #                 print('Waiting 10 seconds for data from X3X...')
        #                 ttime.sleep(10)
        #                 try:
        #                     data[key] = xs_det._datum_ids[idx]
        #                 except IndexError:
        #                     print('WARNING! X3X did not receive all the pulses!')
        #                     print('         Continuing...')
        #                     break  # It won't find anymore data so might as well break


        now = ttime.time()
        current_scan = self.flying_dev.parameters.current_scan.get()
        yield {
            'data': {
                'energy': direction,
                'i0_time': i0_time,
                'i0': i0,
                'im': im,
                'it': it,
                **{f'xs_id_mono_fly_channel{j+1:02d}': self.xs_detectors[0]._datum_ids[j] for j in range(8)}
            },
            'timestamps': {
                'energy': now,
                'i0_time': now,
                'i0': now,
                'im': now,
                'it': now,
                **{f'xs_id_mono_fly_channel{j+1:02d}': now for j in range(8)}
            },
            'time': now,
            'seq_num': 1,
            'filled': {f'xs_id_mono_fly_channel{j+1:02d}': False for j in range(8)},
            'descriptor': f"scan_{current_scan:03d}",
        }

            # print(f"{print_now()}: current_scan: {current_scan}")

            # yield {
            #     'data': data,
            #     'timestamps': timestamps,
            #     'time': now,
            #     'seq_num': ii,
            #     'filled': filled,
            #     'descriptor': 'scan_000',
            # }

        print(f"{print_now()}: after docs emitted in collect")


    def collect_asset_docs(self):
        print(f"{print_now()}: before collecting asset docs from xs in collect_asset_docs")
        for xs_det in self.xs_detectors:
            yield from xs_det.collect_asset_docs()
        print(f"{print_now()}: after collecting asset docs from xs in collect_asset_docs")

    def stop(self):
        # I don't think this is running on stop :-(
        print(f"{print_now()}: I am running the stop function...")
        self._continue_after_pausing = False

        # Abort any active scan
        if self.control.scan_in_progress.get() == 1:
            if self.control.abort.write_access is True:
                print('Aborting any active scans...')
                yield from abs_set(self.control.abort, 1)

        # Disable flying mode
        if self.flying_dev.control.control.write_access is True:
           print('Disabling fly mode...')
           yield from abs_set(self.flying_dev.control.control, 0)

        # unstage flyer
        print('Unstaging the flyer...')
        yield from self.unstage()

        #reset scaler count mode
        print('Resetting the scaler...')
        yield from mv(sclr1.count_mode, 1)
        sclr1.read_attrs = ["channels.chan2", "channels.chan3", "channels.chan4"]
        print(f"{print_now()}: I am leaving the stop function...")
        pass

    def abort(self):
        self.stop()

    def update_lut(self):
        # Make some assumptions:
        N = 20  # Number of points in LUT
        # Define the "knot" values from the calibration
        u_cal = energy.utoelookup.t
        e_cal = energy.etoulookup.t
        Umin = min(u_cal)
        Umax = max(u_cal)

        # Define an easy variable to call
        u2e = energy.utoelookup

        # Make new PV values
        uRBV = np.linspace(Umin, Umax, N, dtype=float)
        eRBV = u2e(uRBV).astype(float)

        # Output
        if self.lut_u.write_access:
            self.lut_u.put(uRBV*1000)
        else:
            print(f"No write access to LUT-Gap\n{uRBV=}")
        if self.lut_e.write_access:
            self.lut_e.put(eRBV*1000)
        else:
            print(f"No write access to LUT-Energy\n{eRBV=}")



def setup_zebra_for_xas(flyer):
    # Common stage_sigs
    ## PC Tab
    # Setup
    flyer.stage_sigs[flyer.zebra.pc.data.cap_enc1_bool] = 1
    flyer.stage_sigs[flyer.zebra.pc.data.cap_enc2_bool] = 1
    flyer.stage_sigs[flyer.zebra.pc.data.cap_enc3_bool] = 1
    flyer.stage_sigs[flyer.zebra.pc.data.cap_enc4_bool] = 0
    # flyer.stage_sigs[flyer.zebra.pc.enc] = 0
    # flyer.stage_sigs[flyer.zebra.pc.dir] = 0
    # flyer.stage_sigs[flyer.zebra.pc.tspre] = 1
    ## ENC tab
    flyer.stage_sigs[flyer.zebra.pc.enc_pos1_sync] = 1
    flyer.stage_sigs[flyer.zebra.pc.enc_pos2_sync] = 1
    flyer.stage_sigs[flyer.zebra.pc.enc_pos3_sync] = 1
    flyer.stage_sigs[flyer.zebra.pc.enc_pos4_sync] = 1
    ## SYS tab
    # flyer.stage_sigs[flyer.zebra.output1.ttl.addr] = 52  # PULSE_1 --> TTL1 --> xs
    # flyer.stage_sigs[flyer.zebra.output2.ttl.addr] = 52  # PULSE_1 --> TTL2 --> merlin
    # flyer.stage_sigs[flyer.zebra.output3.ttl.addr] = 36  # OR1 --> TTL3 --> scaler
    # flyer.stage_sigs[flyer.zebra.output4.ttl.addr] = 52  # PULSE_1 --> TTL4 --> dexela
    flyer.stage_sigs[flyer.zebra.output1.ttl.addr] = 1  # IN1_TTL --> TTL1 --> xs
    flyer.stage_sigs[flyer.zebra.output2.ttl.addr] = 1  # IN1_TTL --> TTL2 --> merlin
    flyer.stage_sigs[flyer.zebra.output3.ttl.addr] = 36  # OR1 --> TTL3 --> scaler
    flyer.stage_sigs[flyer.zebra.output4.ttl.addr] = 1  # IN1_TTL --> TTL4 --> dexela

    ## Specific stage sigs for Zebra - XAS_FLY
    # PC Tab
    # Arm
    flyer.stage_sigs[flyer.zebra.pc.trig_source] = 0
    # Gate
    flyer.stage_sigs[flyer.zebra.pc.gate_source] = 1  # 0 = Position, 1 = Time
    # flyer.stage_sigs[flyer.zebra.pc.gate_start] = 0
    # flyer.stage_sigs[flyer.zebra.pc.gate_width] = 10
    # flyer.stage_sigs[flyer.zebra.pc.gate_step] = 10.1
    flyer.stage_sigs[flyer.zebra.pc.gate_num] = 1
    # Pulse
    flyer.stage_sigs[flyer.zebra.pc.pulse_source] = 0  # 0 = Position, 1 = Time
    # flyer.stage_sigs[flyer.zebra.pc.pulse_start] = 0
    # flyer.stage_sigs[flyer.zebra.pc.pulse_width] = 0.9
    # flyer.stage_sigs[flyer.zebra.pc.pulse_step] = 1
    # flyer.stage_sigs[flyer.zebra.pc.pulse_max] = 10
    ## OR Tab
    flyer.stage_sigs[flyer.zebra.or1.use1] = 1  # 0 = No, 1 = Yes
    flyer.stage_sigs[flyer.zebra.or1.use2] = 1
    flyer.stage_sigs[flyer.zebra.or1.use3] = 0
    flyer.stage_sigs[flyer.zebra.or1.use4] = 0
    flyer.stage_sigs[flyer.zebra.or1.input_source1] = 54
    flyer.stage_sigs[flyer.zebra.or1.input_source2] = 55
    flyer.stage_sigs[flyer.zebra.or1.input_source3] = 53
    flyer.stage_sigs[flyer.zebra.or1.input_source4] = 0
    flyer.stage_sigs[flyer.zebra.or1.invert1] = 0  # 0 = No, 1 = Yes
    flyer.stage_sigs[flyer.zebra.or1.invert2] = 0
    flyer.stage_sigs[flyer.zebra.or1.invert3] = 0
    flyer.stage_sigs[flyer.zebra.or1.invert4] = 0
    ## PULSE Tab
    flyer.stage_sigs[flyer.zebra.pulse1.input_addr] = 0
    flyer.stage_sigs[flyer.zebra.pulse1.input_edge] = 0  # 0 = rising, 1 = falling
    flyer.stage_sigs[flyer.zebra.pulse1.delay] = 0.0
    # flyer.stage_sigs[flyer.zebra.pulse1.width] = 0.1  # Written by plan
    flyer.stage_sigs[flyer.zebra.pulse1.time_units] = 's'
    # flyer.stage_sigs[flyer.zebra.pulse2.input_addr] = 30
    # flyer.stage_sigs[flyer.zebra.pulse2.input_edge] = 0  # 0 = rising, 1 = falling
    # flyer.stage_sigs[flyer.zebra.pulse2.delay] = 0
    # flyer.stage_sigs[flyer.zebra.pulse2.width] = 0.1
    # flyer.stage_sigs[flyer.zebra.pulse2.time_units] = 0
    flyer.stage_sigs[flyer.zebra.pulse3.input_addr] = 1  # IN1_TTL, not PULSE1 or 52
    flyer.stage_sigs[flyer.zebra.pulse3.input_edge] = 0  # 0 = rising, 1 = falling
    flyer.stage_sigs[flyer.zebra.pulse3.delay] = 0.0
    flyer.stage_sigs[flyer.zebra.pulse3.width] = 0.1
    flyer.stage_sigs[flyer.zebra.pulse3.time_units] = 'ms'

    flyer.stage_sigs[flyer.zebra.pulse4.input_addr] = 1  # IN1_TTL, not PULSE1 or 52
    flyer.stage_sigs[flyer.zebra.pulse4.input_edge] = 1  # 0 = rising, 1 = falling
    flyer.stage_sigs[flyer.zebra.pulse4.delay] = 0
    flyer.stage_sigs[flyer.zebra.pulse4.width] = 0.1
    flyer.stage_sigs[flyer.zebra.pulse4.time_units] = 'ms'


flyer_id_mono = FlyerIDMono(flying_dev=id_fly_device,
                            zebra=nanoZebra,
                            xs_detectors=[xs_id_mono_fly],
                            scaler=sclr1,
                            pulse_cpt='pulse1',
                            pulse_width=0.05)
setup_zebra_for_xas(flyer_id_mono)


# Helper functions for quick vis:
def plot_flyer_id_mono_data(uid_or_scanid, e_min=None, e_max=None, fname=None, root='/home/xf05id1/current_user_data/', num_channels=7, plot=True):
    hdr = db[uid_or_scanid]
    stream_names = hdr.stream_names
    stream_names.remove('baseline')

    if (plot):
        fig, ax = plt.subplots()


    for stream in sorted(stream_names):
        if 'monitor' in stream:
            continue
        tbl = hdr.table(stream_name=stream)

        if (e_min is None):
            e_min = xs.channel1.rois.roi01.bin_low.get()
        if (e_max is None):
            e_max = xs.channel1.rois.roi01.bin_high.get()

        fname = f"scan{hdr.start['scan_id']}_{stream}.txt"
        fname = root + fname

        d = []
        for i in range(num_channels):
            d.append(np.array(list(hdr.data(f'xs_id_mono_fly_channel{i+1:02}', stream_name=stream)))[:, e_min:e_max].sum(axis=1))
        d = np.array(d)

        i0 = np.array(tbl['i0'])
        energy = np.array(tbl['energy'])

        spectrum_unnormalized = d.sum(axis=0)
        spectrum = spectrum_unnormalized / i0

        res = np.vstack((energy, i0, spectrum_unnormalized, spectrum))

        if (plot):
            ax.plot(energy, spectrum, label=stream)
            ax.set(xlabel='Energy [eV]', ylabel='Normalized Spectrum [Arb]')
        np.savetxt(fname, res.T)

    if (plot):
        ax.legend()
    return res

# Export data function
def export_flyer_id_mono_data(uid_or_scanid, roi=1, e_min=None, e_max=None, fname=None, root='/home/xf05id1/test_flyingID/', flyer=flyer_id_mono):
    hdr = c[uid_or_scanid]
    scan_streams = list(hdr)
    scan_streams.remove('baseline')
    scan_streams = [s for s in scan_streams if "monitor" not in s]

    # Assuming first detector in the xs_detectors list has the correct ROI in ROI1
    # TODO: This should not look at the object! This should be stored in metadata
    if (e_min is None):
        e_min = getattr(flyer.xs_detectors[0].channel01, f"mcaroi{roi:02}").min_x.get()
        # e_min = flyer.xs_detectors[0].channel01.mcaroi01.min_x.get()
    if (e_max is None):
        e_max = getattr(flyer.xs_detectors[0].channel01, f"mcaroi{roi:02}").min_x.get() + getattr(flyer.xs_detectors[0].channel01, f"mcaroi{roi:02}").size_x.get()
        # e_max = flyer.xs_detectors[0].channel01.mcaroi01.min_x.get() + flyer.xs_detectors[0].channel01.mcaroi01.size_x.get()

    ring_current_start = str(hdr["baseline"]["data"]["ring_current"][0].round(2))

    staticheader = f"# XDI/1.0 MX/2.0\n" \
                 + f"# Beamline.name: {hdr.start['beamline_id']}\n" \
                 + f"# Facility.name: NSLS-II\n" \
                 + f"# Facility.ring_current: {ring_current_start}\n" \
                 + f"# IVU.harmonic: {hdr.start['scan']['harmonic']}\n" \
                 + f"# Mono.name: Si 111\n" \
                 + f"# Scan.start.uid: {hdr.start['uid']}\n" \
                 + f"# Scan.start.scanid: {hdr.start['scan_id']}\n" \
                 + f"# Scan.start.time: {hdr.start['time']}\n" \
                 + f"# Scan.start.ctime: {ttime.ctime(hdr.start['time'])}\n" \
                 + f"# Scan.ROI.name: {hdr.start['scan']['roi_names'][roi-1]}\n" \
                 + f"# Scan.ROI.number: {roi}\n" \
                 + f"# Scan.ROI.range: {f'[{e_min}:{e_max}]'}\n" \
                 + f"# \n"

    for stream in sorted(scan_streams):
        fname = f"scan_{hdr.start['scan_id']}_{stream}.txt"
        fname = root + fname

        print(f'{stream}')
        print(f"  Export to {fname}")
        print(f'  Collecting data...')
        # Get the full table
        tbl = hdr[stream]["data"]
        df = pd.DataFrame()
        keys = [k for k in tbl.keys()[:] if "time" not in k]
        for k in keys:
            print(f"{k=}")
            if "channel" in k:
                # We will process later
                continue
            df[k] = tbl[k].read()[0]

        print(f'  Processing data...')
        print(f"{df=}")
        df.set_index("energy", drop=True, inplace=True)

        ch_names = [ch for ch in keys if "channel" in ch]
        for ch in ch_names:
            df[ch] = np.sum(tbl[ch].read()[0, :, e_min:e_max], axis=1)
            df.rename(columns={ch : ch.split('_')[-1]}, inplace=True)
        df['ch_sum'] = df[[ch for ch in df.keys() if "channel" in ch]].sum(axis=1)

        # Prepare for export
        col_names = [df.index.name] + list(df.columns)
        for i, col in enumerate(col_names):
            staticheader += f"# Column {i+1:02}: {col}\n"
        staticheader += "# \n# "

        print('  Exporting data...')
        # Export data to file
        with open(fname, 'w') as f:
            f.write(staticheader)
        df.to_csv(fname, float_format="%.3f", sep=' ', mode='a')

        print('  Complete!')

def flying_xas(num_passes=1, shutter=True, md=None):
    v = flyer_id_mono.flying_dev.parameters.speed.get()
    w = flyer_id_mono.flying_dev.parameters.trigger_width.get()
    dt = w / v
    flyer_id_mono.pulse_width = dt
    yield from check_shutters(shutter, 'Open')
    yield from bp.fly([flyer_id_mono])
    # yield from fly_multiple_passes([flyer_id_mono], num_passes=num_passes,
    #                                md=md, shutter=shutter)
    yield from check_shutters(shutter, 'Close')


@append_srx_kwargs_md
@parameter_annotation_decorator({
    "parameters": {
        "flyers": {"default": "['flyer_id_mono']"},
    }
})
def fly_multiple_passes(e_start, e_stop, e_width, dwell, num_pts, *,
                        num_scans=1, scan_type='uni', shutter=True, plot=False,
                        flyers=[flyer_id_mono], harmonic=1, roi_num=1,
                        md=None):
    """This is a modified version of bp.fly to support multiple passes of the flyer."""
    flyer_id_mono.flying_dev.parameters.first_trigger.put(e_start)
    flyer_id_mono.flying_dev.parameters.last_trigger.put(e_stop)
    flyer_id_mono.flying_dev.parameters.trigger_width.put(e_width)
    flyer_id_mono.flying_dev.parameters.num_triggers.put(num_pts)
    flyer_id_mono._traj_info['num_triggers'] = num_pts
    flyer_id_mono._traj_info['energy_start'] = e_start
    flyer_id_mono._traj_info['energy_stop'] = e_stop
    flyer_id_mono._traj_info['energy_width'] = e_width

    if 'uni' in scan_type.lower():
        scan_type = 'unidirectional'
        flyer_id_mono.flying_dev.control.scan_type.put(0)
    elif 'bi' in scan_type.lower():
        scan_type = 'bidirectional'
        flyer_id_mono.flying_dev.control.scan_type.put(1)
    else:
        raise ValueError(f'Unknown scan type! {scan_type}')

    v = e_width / dwell
    flyer_id_mono.flying_dev.parameters.speed.put(v)
    e_step = (e_stop - e_start) / (num_pts- 1)
    dt = e_width / v

    if (abs(e_step) <= e_width):
        raise ValueError('Cannot have energy collection widths larger than energy step!')

    # Get a harmonic value (assuming no detuning)
    if harmonic < 3:
        harmonic = 3
        # _, _, ugap = energy.energy_to_positions(e_start/1000, harmonic, 0)
        _, _, ugap = energy.energy_to_positions(e_stop/1000, harmonic, 0)
        while True:
            # _, _, ugap = energy.energy_to_positions(e_start/1000, harmonic+2, 0)
            _, _, ugap = energy.energy_to_positions(e_stop/1000, harmonic+2, 0)
            # if ugap < energy.u_gap.low_limit:
            if ugap < np.amin(flyer_id_mono.lut_u.get()):
                break
            harmonic += 2

    # set harmonic
    flyer_id_mono.flying_dev.parameters.harmonic.put(harmonic)

    md = get_stock_md(md)
    md['scan']['type'] = 'XAS_FLY'
    # md['scan']['energy'] = list(np.linspace(e_start, e_stop, num=num_pts))
    md['scan']['num_points'] = num_pts
    md['scan']['scan_input'] = [e_start, e_stop, e_width, dwell, num_pts]
    md['scan']['sample_name'] = ''
    md['scan']['dwell'] = dwell
    md['scan']['num_scans'] = num_scans
    md['scan']['harmonic'] = harmonic
    md['scan']['roi_num'] = roi_num
    md['scan']['direction'] = scan_type
    # put in try-except since this will not work with QD IOC
    try:
        ch = flyers[0].xs_detectors[0].channel01
        md['scan']['roi_names'] = [ch.get_mcaroi(mcaroi_number=i+1).roi_name.get()
                                   for i in range(ch.get_mcaroi_count())]
    except:
        pass

    d, dets = [], []
    for fly in flyers:
        for flying_xs in fly.xs_detectors:
            d.append(flying_xs.name)
            dets.append(flying_xs)
            yield from bps.mv(flying_xs.fly_next, True)
    # md['scan']['detectors'] = d
    get_det_md(md, dets)

    livepopup = []
    roi_pv = flyer_id_mono.xs_detectors[0].channel01.mcaroi01.ts_total
    if plot is True:
        yield from mov(flyer_id_mono.xs_detectors[0].channel01.mcaroi.ts_control, 2, settle_time=0.1, timeout=1)
        yield from mov(flyer_id_mono.xs_detectors[0].channel01.mcaroi.ts_num_points, num_pts, settle_time=0.1, timeout=1)
        yield from mov(flyer_id_mono.xs_detectors[0].channel01.mcaroi.ts_control, 0, settle_time=0.1, timeout=1)

        livepopup = [SRX1DTSFlyerPlot(roi_pv.name,
                                      xstart=e_start,
                                      xstep=(e_stop-e_start) / (num_pts-1),
                                      xlabel="Energy (eV)")]


    # Set the currect pulse length for zebra1 - microZebra
    yield from bps.mov(microZebra.pulse1.width, dwell, timeout=3)
    yield from bps.mov(microZebra.pulse2.width, dwell, timeout=3)

    @subs_decorator(livepopup)
    @ts_monitor_during_decorator([roi_pv])
    def plan():
        # yield from check_shutters(shutter, 'Open')
        uid = yield from bps.open_run(md)
        yield from mv(sclr1.count_mode, 0)
        yield from mv(energy, e_start)
        # Wait for scaler to be "done"
        yield from bps.sleep(2)  # TODO: THIS SHOULD BE SMARTER!
        yield from bps.mov(sclr1.erase_all, 1)
        # print(f"Kickoff: {flyers}")
        for flyer in flyers:
            # print(f"  Kicking off {flyer}...")
            flyer.pulse_width = dwell
            yield from bps.mv(flyer.flying_dev.parameters.num_scans, num_scans)
            yield from bps.kickoff(flyer, wait=True)
        for n in range(num_scans):
            print(f"\n\n*** {print_now()} Iteration #{n+1} ***\n")
            yield from bps.checkpoint()
            if shutter is True:
                yield from abs_set(shut_d.request_open, 1, wait=True, timeout=2)
            # flyer_id_mono.scaler.erase_start.put(1)
            for flyer in flyers:
                # print(f"  {flyer.name} complete...")
                yield from bps.complete(flyer, wait=True)
            if shutter is True:
                yield from abs_set(shut_d.request_open, 0, wait=False)
            for flyer in flyers:
                # print(f"  {flyer} collect...", end='', flush=True)
                yield from bps.collect(flyer)
                # print("done")
            # This is a work around until the flyer can be rewritten
            yield Msg("nuke_the_cache", flyer_id_mono)
        yield from check_shutters(shutter, 'Close')
        yield from mv(sclr1.count_mode, 1)
        yield from bps.close_run()
        for flyer in flyers:
            # yield from bps.mv(flyer.flying_dev.control, "disable")
            st = flyer.flying_dev.control.set("disable")
            try:
                st.wait(10)
            except WaitTimeoutError as ex:
                print("Timeout while disabling control of IVU!")
                print("Trying again...")

                print("  disabling...", end="", flush=True)
                st = self.flying_dev.control.set("disable")
                st.wait(10)
                print("done")
            except Exception as ex:
                raise ex

        return uid


    return (yield from plan())

def flying_xas_reset():
    # Abort any active scan
    if id_fly_device.control.scan_in_progress.get() == 1:
        if id_fly_device.control.abort.write_access is True:
            print('Aborting any active scans...')
            yield from mov(id_fly_device.control.abort, 1, timeout=10)

    # Disable flying mode
    if flyer_id_mono.flying_dev.control.control.write_access is True:
       print('Disabling fly mode...')
       st = id_fly_device.control.set("disable")
       st.wait(timeout=10)

    # Unstage flyer
    print('Unstaging the flyer...')
    yield from unstage(flyer_id_mono)

    # Reset scaler count mode
    print('Reinitializing the scaler...')
    sclr1 = SRXScaler("XF:05IDD-ES:1{Sclr:1}", name="sclr1")
    sclr1.read_attrs = ["channels.chan2", "channels.chan3", "channels.chan4"]
    yield from mov(sclr1.count_mode, 1, timeout=10)


def reset_after_flying_xas():
    ivu_sp = EpicsSignal("SR:C5-ID:G1{IVU21:1-Ax:Gap}-Mtr-SP")
    ivu_rb = EpicsSignal("SR:C5-ID:G1{IVU21:1-Ax:Gap}-Mtr.RBV")
    ivu_move = EpicsSignal("SR:C5-ID:G1{IVU21:1-Ax:Gap}-Mtr-Go")

    ivu_energy = energy.undulator_energy(harmonic=energy.selected_harmonic.get())
    bragg_energy = energy.position.energy

    if np.abs(ivu_energy - bragg_energy) > 0.005:  # keV
        print(f"Undulator energy: {ivu_energy:3.3f} keV")
        print(f"Bragg energy:     {bragg_energy:3.3f} keV")

        _, _, sp = energy.energy_to_positions(bragg_energy, energy.selected_harmonic.get(), 0)

        def cb_gap_move(*, value, old_value, **kwargs):
            value = sp
            old_value = ivu_rb.get()
            if np.abs(value - old_value) < 3:
                return True
            return False

        st = SubscriptionStatus(ivu_rb, cb_gap_move, run=True)
        yield from mov(ivu_sp, sp, timeout=5)
        yield from mov(ivu_move, 1)

        try:
            st.wait(timeout=10)
        except WaitTimeoutError:
            print("Timeout error!")

"""
TODO: All scan directions and modes (uni/bi-directional), DONE
TODO: setup stage_sigs for scaler
TODO: Monitor and LivePlot of data
        need to cleanup LivePlot code
        need to label each pass with different line (and color)
TODO: Unstage the detectors (xs, scaler) on RE.abort()
TODO: Compare number of triggers from Zebra's data collected table with actual number of emitted pulses.
TODO: Use timestamps from Zebra's data collected table to generate the events.
TODO: Autodetect element of interest from roi name?
TODO: Need export function, suitcase?
"""
