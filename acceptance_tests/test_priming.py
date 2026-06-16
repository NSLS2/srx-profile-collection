# these are mainly notes of what we tried to test the hdf5 detector priming
# issue
from ophyd.sim import motor1, motor2


RE.clear_suspenders()

# here is what we can use to reproduce the error
#changed the flyer device to be aware of fast vs slow axis in a 2D scan
#should abstract this method to use fast and slow axes, rather than x and y
def scan_and_fly_test(xstart, xstop, xnum, ystart, ystop, ynum, dwell, *,
                 delta=0.002, shutter = True,
                 xmotor=hf_stage.x, ymotor=hf_stage.y,
                 xs=xs, ion=sclr1, align = False,
                 flying_zebra=flying_zebra, md=None):
    """

    Read IO from SIS3820.
    Zebra buffers x(t) points as a flyer.
    Xpress3 is our detector.
    The aerotech has the x and y positioners.
    delta should be chosen so that it takes about 0.5 sec to reach the gate??
    ymotor  slow axis
    xmotor  fast axis
    """
    c2pitch_kill=EpicsSignal("XF:05IDA-OP:1{Mono:HDCM-Ax:P2}Cmd:Kill-Cmd")
    if md is None:
        md = {}
    if delta is None:
        delta=0.002
    yield from abs_set(ymotor, ystart, wait=True) # ready to move
    yield from abs_set(xmotor, xstart - delta, wait=True) # ready to move

    if shutter is True:
        yield from mv(shut_b, 'Open')

    if align == True:
        fly_ps = PeakStats(dcm.c2_pitch.name,i0.name)
        align_scan = scan([sclr1], dcm.c2_pitch, -19.320, -19.360, 41)
        align_scan = bp.subs_wrapper(align_scan,fly_ps)
        yield from align_scan
        yield from abs_set(dcm.c2_pitch,fly_ps.max[0],wait=True)
        #ttime.sleep(10)
        #yield from abs_set(c2pitch_kill, 1)
    else:
        ttime.sleep(0.)

    md = ChainMap(md, {
        'plan_name': 'scan_and_fly',
        'detectors': [zebra.name,xs.name,ion.name],
        'dwell' : dwell,
        'shape' : (xnum,ynum),
        'scaninfo' : {'type': 'XRF_fly', 'raster' : False, 'fast_axis':flying_zebra._fast_axis},
        'scan_params' : [xstart,xstop,xnum,ystart,ystop,ynum,dwell]
        }
    )


    from bluesky.plan_stubs import stage, unstage
    @stage_decorator([xs])
    def fly_each_step(detectors, motor, step, firststep):
        "See http://nsls-ii.github.io/bluesky/plans.html#the-per-step-hook"
        # First, let 'scan' handle the normal y step, including a checkpoint.
        yield from bps.one_1d_step(detectors, motor, step)

        # Now do the x steps.
        v = (xstop - xstart) / (xnum-1) / dwell  # compute "stage speed"
        yield from abs_set(xmotor, xstart - delta, wait=True) # ready to move
        yield from abs_set(xmotor.velocity, v, wait=True)  # set the "stage speed"
        yield from abs_set(xs.hdf5.num_capture, xnum)
        yield from abs_set(xs.settings.num_images, xnum)
        yield from abs_set(ion.nuse_all,xnum)
        # arm the Zebra (start caching x positions)
        yield from kickoff(flying_zebra, xstart=xstart, xstop=xstop, xnum=xnum, dwell=dwell, wait=True)
        yield from abs_set(xs.settings.acquire, 1)  # start acquiring images
        yield from abs_set(ion.erase_start, 1) # arm SIS3820, note that there is a 1 sec delay in setting X into motion
                                               # so the first point *in each row* won't normalize...
        #if firststep == True:
        #    ttime.sleep(0.)
        ttime.sleep(10)
        yield from abs_set(xmotor, xstop+1*delta, wait=True)  # move in x
        yield from abs_set(xs.settings.acquire, 0)  # stop acquiring images
        yield from abs_set(ion.stop_all, 1)  # stop acquiring scaler
        yield from complete(flying_zebra)  # tell the Zebra we are done
        yield from collect(flying_zebra)  # extract data from Zebra
        yield from abs_set(xmotor.velocity, 1.,wait=True)  # set the "stage speed"

    def at_scan(name, doc):
        scanrecord.current_scan.put(doc['uid'][:6])
        scanrecord.current_scan_id.put(str(doc['scan_id']))
        scanrecord.current_type.put(md['scaninfo']['type'])
        scanrecord.scanning.put(True)
        scanrecord.time_remaining.put((dwell*xnum + 3.8)/3600)
    def finalize_scan(name, doc):
        logscan_detailed('xrf_fly')
        scanrecord.scanning.put(False)
        scanrecord.time_remaining.put(0)

    #@subs_decorator([LiveTable([ymotor]), RowBasedLiveGrid((ynum, xnum), ion.name, row_key=ymotor.name), LiveZebraPlot()])
    #@subs_decorator([LiveTable([ymotor]), LiveGrid((ynum, xnum), sclr1.mca01.name)])
    @subs_decorator([LiveTable([ymotor])])
    #@subs_decorator([LiveGrid((ynum, xnum+1), xs.channel1.rois.roi01.value.name,extent=(xstart,xstop,ystop,ystart))])
    @subs_decorator({'start':at_scan})
    @subs_decorator({'stop':finalize_scan})
    @monitor_during_decorator([xs.channel1.rois.roi01.value])  # monitor values from xs
    #@monitor_during_decorator([xs], run=False)  # monitor values from xs
    @stage_decorator([flying_zebra])  # Below, 'scan' stage ymotor.
    @run_decorator(md=md)
    def plan():
        yield from bps.mov(xs.total_points, xnum)
        #yield from abs_set(xs.settings.trigger_mode, 'TTL Veto Only')
        yield from abs_set(xs.external_trig, True)
        ystep = 0
        for step in np.linspace(ystart, ystop, ynum):
            scanrecord.time_remaining.put( (ynum - ystep) * ( dwell * xnum + 3.8 ) / 3600.)
            ystep = ystep + 1
            # 'arm' the xs for outputting fly data
            yield from abs_set(xs.hdf5.fly_next, True)
#            print('h5 armed\t',time.time())
            if step == ystart:
                firststep = True
            else:
                firststep = False
            yield from fly_each_step([], ymotor, step, firststep)
#            print('return from step\t',time.time())
        yield from abs_set(xs.external_trig, False)
        yield from abs_set(ion.count_mode, 1)
        if shutter is True:
            yield from mv(shut_b, 'Close')


    return (yield from plan())


'''
    fast axis : xmotor
        xstart, xstop, num : fast axis
'''

xmotor = motor1
xmotor.velocity = Signal(name="xmotor_velocity")
ymotor = motor1
ymotor.velocity = Signal(name="ymotor_velocity")

# fast motor
xstart = 0
xstop = 1
xnum = 2001
# dwell time?
dwell = .01

# slow motor : dummy values
# ynum will be the number of times to move in slow axis
ystart = 0
ystop= 1
ynum = 1


from functools import partial
plan_test=partial(scan_and_fly_test,xstart=xstart, xstop=xstop, xnum=xnum, ystart=ystart, ystop=ystop, ynum=ynum, dwell=dwell,
                  delta=.002, xmotor=xmotor, ymotor=ymotor, xs=xs, ion=sclr1, align=False,
                  flying_zebra=flying_zebra, shutter=False)


def prime_plan(N, acqtime=.001):
    '''
        This fixes the issue
    '''
    # N : number of points you want to count up to
    yield from bps.abs_set(xs.external_trig, False)
    yield from bps.abs_set(xs.settings.acquire_time, acqtime)
    yield from bps.abs_set(xs.total_points, N)
    yield from bps.stage(xs.hdf5)
    # unset capture so that unstage doesn't hang a bit
    yield from bps.abs_set(xs.hdf5.capture, 0)
    yield from bps.unstage(xs.hdf5)

def prime_xs_hdf5(N):
    print(f"Priming the hdf5 plugin to capture at least {N} images")
    xs.external_trig.put(False)
    xs.settings.acquire_time.put(acqtime)
    xs.total_points.put(N)
    xs.hdf5.stage()
    # unset capture so that unstage doesn't hang a bit
    xs.hdf5.capture.put(0)
    xs.hdf5.unstage()

def init_hdf5_plugin():
    xs.hdf5.file_name.put("deleteme")
    xs.hdf5.capture.put(1)
    xs.hdf5.capture.put(0)
    # we usually expect 1000 frames
    prime_xs_hdf5(1000)

'''
old
def prime_plan(N, acqtime=.001):
    # N : number of points you want to count up to
    yield from bps.abs_set(xs.external_trig, False)
    yield from bps.abs_set(xs.settings.acquire_time, acqtime)
    yield from bps.abs_set(xs.total_points, N)
    yield from bps.stage(xs)
    #for i in range(N):
        #yield from bps.trigger(xs)
    yield from bps.unstage(xs)
'''

# to reset the zebra?
#zebra.pc.block_state_reset.put(1)

# testing the warmup
def init_hdf5_plugin():
    '''
        We need to save a file once
        Not using this right now; it doesn't work
        We need to manually do this for now.
    '''
    xs.external_trig.put(False)
    xs.stage()
    ttime.sleep(.2)
    xs.hdf5.capture.put(1)
    xs.hdf5.capture.put(0)
    xs.unstage()
    xs.hdf5.file_name.put("deleteme")
    #ttime.sleep(.1)
    #xs.hdf5.capture.put(0)
    # we usually expect 1000 frames

'''
Start a flyer:


stage:
zebra.pc.gate_start
zebra.pc.gate_width
zebra.pc.gate_step
zebra.pc.gate_num

zebra.pc.pulse_start
zebra.pc.pulse_width
zebra.pc.pulse_step
zebra.pc.pulse_max


kickoff:
zebra.pc.arm

complete:
(wait on)
zebra.pc.armed

collect


# detector:

'''
