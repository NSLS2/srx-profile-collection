print(f'Loading {__file__}...')


from ophyd import EpicsSignal, EpicsSignalRO, Device
from ophyd import Component as Cpt


class SRXScanRecord(Device):

    class OneScan(Device):
        p1s = Cpt(EpicsSignal, 'P1-S')
        p2s = Cpt(EpicsSignal, 'P2-S')
        p1i = Cpt(EpicsSignal, 'P1-I')
        p2i = Cpt(EpicsSignal, 'P2-I')
        p1stp = Cpt(EpicsSignal, 'P1-STP')
        p2stp = Cpt(EpicsSignal, 'P2-STP')
        p1npts = Cpt(EpicsSignalRO, 'P1-NPTS')
        p2npts = Cpt(EpicsSignalRO, 'P2-NTPS')
        p1e = Cpt(EpicsSignalRO, 'P1-E')
        p2e = Cpt(EpicsSignalRO, 'P2-E')
        p1ena = Cpt(EpicsSignal, 'P1-ENA')
        p2ena = Cpt(EpicsSignal, 'P2-ENA')
        curpt = Cpt(EpicsSignal, 'CURPT')
        npts = Cpt(EpicsSignalRO, 'NPTS')
        tpp = Cpt(EpicsSignal, 'TPP')
        ena = Cpt(EpicsSignal, 'ENA')
        acq = Cpt(EpicsSignal, 'ACQ')
        e1s = Cpt(EpicsSignal, 'E1-S')
        e1i = Cpt(EpicsSignal, 'E1-I')
        e1npts = Cpt(EpicsSignal, 'E1-NPTS')
        e1e = Cpt(EpicsSignal, 'E1-E')
        e2s = Cpt(EpicsSignal, 'E2-S')
        e2i = Cpt(EpicsSignal, 'E2-I')
        e2npts = Cpt(EpicsSignal, 'E2-NPTS')
        e2e = Cpt(EpicsSignal, 'E2-E')
        e3s = Cpt(EpicsSignal, 'E3-S')
        e3i = Cpt(EpicsSignal, 'E3-I')
        e3npts = Cpt(EpicsSignal, 'E3-NPTS')
        e3e = Cpt(EpicsSignal, 'E3-E')
        efs = Cpt(EpicsSignal, 'EF-S')
        Eena = Cpt(EpicsSignal, 'E-ENA')
        Ewait = Cpt(EpicsSignal, 'E-WAIT')
        filename = Cpt(EpicsSignal, 'FILEN')
        sampname = Cpt(EpicsSignal, 'SAMPN')
        roi = Cpt(EpicsSignal, 'ROIN')
        detune = Cpt(EpicsSignal, 'DETUNE')

#    scans = [ Cpt(OneScan,'Scan'+str(i)+':') for i in range(0,8) ]
    scan0 = Cpt(OneScan, 'Scan0:')
    scan1 = Cpt(OneScan, 'Scan1:')
    scan2 = Cpt(OneScan, 'Scan2:')
    scan3 = Cpt(OneScan, 'Scan3:')
    scan4 = Cpt(OneScan, 'Scan4:')
    scan5 = Cpt(OneScan, 'Scan5:')
    scan6 = Cpt(OneScan, 'Scan6:')
    scan7 = Cpt(OneScan, 'Scan7:')
    scan8 = Cpt(OneScan, 'Scan8:')
    scan9 = Cpt(OneScan, 'Scan9:')
    scan10 = Cpt(OneScan, 'Scan10:')
    scan11 = Cpt(OneScan, 'Scan11:')
    scan12 = Cpt(OneScan, 'Scan12:')
    scan13 = Cpt(OneScan, 'Scan13:')
    scan14 = Cpt(OneScan, 'Scan14:')
    scan15 = Cpt(OneScan, 'Scan15:')

    def cp(self, src, dest):
        '''
        Copy all elements of the scan object from src to dest.
        src and dest must be the string names of the scans
        scans are indexed starting at 0
        '''
        for i in ['p1s', 'p2s', 'p1i', 'p2i', 'p1stp', 'p2stp',
                  'p1ena', 'p2ena', 'curpt', 'tpp', 'ena', 'acq',
                  'e1s', 'e1i', 'e1npts', 'e1e', 'e2s', 'e2i', 'e2npts', 'e2e',
                  'e3s', 'e3i', 'e3npts', 'e3e', 'efs', 'Eena', 'Ewait',
                  'filename', 'sampname', 'roi', 'detune']:
            getattr(getattr(self, dest), i).put(getattr(getattr(scanrecord, src), i).get())

    def cp_XANES(self, src_num, dest_num):
        '''
        Copy all energy elements of scan number src_num to scan number dest_num
        scan numbers are indexed starting at 1
        '''
        src = 'scan{}'.format(src_num-1)
        dest = 'scan{}'.format(dest_num-1)
        for i in ['p1s', 'p2s', 'p1i', 'p2i', 'p1stp', 'p2stp',
                  'p1ena', 'p2ena', 'curpt', 'tpp', 'ena', 'acq',
                  'e1s', 'e1i', 'e1npts', 'e1e', 'e2s', 'e2i', 'e2npts', 'e2e',
                  'e3s', 'e3i', 'e3npts', 'e3e', 'efs', 'Eena', 'Ewait',
                  'filename', 'sampname', 'roi', 'detune']:
            pass
            getattr(getattr(self, dest), i).put(getattr(getattr(scanrecord, src), i).get())

    def cp_XRF(self, src_num, dest_num):
        '''
        Copy all positional elements of scan number src_num to scan number
        dest_num scan numbers are indexed starting at 1
        '''
        src = 'scan{}'.format(src_num-1)
        dest = 'scan{}'.format(dest_num-1)
        for i in ['p1s', 'p2s', 'p1i', 'p2i', 'p1stp', 'p2stp',
                  'p1ena', 'p2ena', 'curpt', 'tpp', 'ena', 'acq',
                  'sampname']:
            getattr(getattr(self, dest), i).put(getattr(getattr(scanrecord, src), i).get())

    def disable_scans(self):
        for i in range(0, 16):
            scan = 'scan{}'.format(i)
            getattr(getattr(scanrecord, scan), 'ena').put(0)
            getattr(getattr(scanrecord, scan), 'Eena').put(0)

    def update_metadata(self):
        md = RE.md['proposal']
        # print(self.cycle.pvname)
        # print(self.cycle.connected)
        # if (self.cycle.connected is True):
        if True:
            self.proposal_num.put(str(md['proposal_id']))
            self.proposal_title.put(str(md['title']))
            # self.SAF.put(str(md['saf_num']))
            self.PI.put(str(md['pi_name']))
            self.cycle.put(str(RE.md['cycle']))
            print('Scanrecord updated.')
        else:
            print('Scanrecord NOT updated...')

    current_scan = Cpt(EpicsSignal, 'Scan:CUR')
    current_scan_id = Cpt(EpicsSignal, 'Scan:CUR_ID')
    current_type = Cpt(EpicsSignal, 'Scan:TYPE')
    time_remaining = Cpt(EpicsSignal, 'Scan:REMTIME')
    scanning = Cpt(EpicsSignal, 'Scan:ENA')
    time_rem_str = Cpt(EpicsSignal, 'Scan:REMTIMESTR')

    proposal_num = Cpt(EpicsSignal, 'PROPOSAL_NUM')
    # If len(string) > 40 characters, then it needs to be a waveform on the IOC
    proposal_title = Cpt(EpicsSignal, 'PROPOSAL_TITLE', string=True)
    SAF = Cpt(EpicsSignal, 'SAF_NUM')
    PI = Cpt(EpicsSignal, 'PI_NAME')
    cycle = Cpt(EpicsSignal, 'CYCLE')

scanrecord = SRXScanRecord('XF:05IDA-CT{IOC:ScanBroker02}',
                           name='scanrecord')

# Set defaults
try:
    scanrecord.scan0.p1s.put(25.0)
    scanrecord.scan0.p2s.put(20.0)
    scanrecord.scan0.p1i.put(0.010)
    scanrecord.scan0.p2i.put(0.010)
    scanrecord.scan0.p1stp.put(3)
    scanrecord.scan0.p2stp.put(3)
    scanrecord.scan0.ena.put(1)
    scanrecord.scan0.acq.put(1)
except:
    print('\nError connecting to scanrecord...\n')
