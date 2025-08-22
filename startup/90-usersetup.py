print(f'Loading {__file__}...')

import os
import time as ttime
import shutil
# import httpx

### Proposal information put into the metadata
# proposal_num = None
# proposal_title = None
# PI_lastname = None
# saf_num = None

# proposal_num = 312933
# data_session = RE.md["data_session"]
# proposal_num = data_session.split("-")[-1]

# # proposal_title = 'SRX Beamline Commissioning'
# # PI_lastname = 'Kiss'
# # saf_num = 311284

# nslsii_api_client = httpx.Client(
#     base_url="https://api.nsls2.bnl.gov"
# )

# proposal_response = nslsii_api_client.get(
#     f"/v1/proposal/{proposal_num}"
# )
# proposal_response.raise_for_status()
# proposal = proposal_response.json()

# PI_lastname = "whoami"
# for user in proposal["users"]:
#     if user["is_pi"]:
#         PI_lastname = user["last_name"]
#         break

# saf_num = proposal["safs"][0]["saf_id"]

# md_proposal = {
#     'proposal_num' :  proposal_num,
#     'proposal_title' : proposal["title"],
#     'PI_lastname' : PI_lastname,
#     'saf_num' : saf_num,
#     'cycle' : proposal["cycles"][-1],
# }

# # proposal_num = 313507
# # proposal_title = 'Data security at SRX'
# # PI_lastname = 'Kiss'
# # saf_num = 312779



# cycle = '2024_cycle1'

# # Set user data in bluesky
# # RE.md['data_session'] = data_session
# RE.md['proposal'] = md_proposal

if if_touch_beamline():
    # Set user data in scanbroker
    scanrecord.update_metadata()

# User data directory and simple scripts
# userdatadir = '/nsls2/data/srx/legacy/xf05id1/experiments/' + str(cycle) + '/' + str(saf_num) + '_' + str(PI_lastname) + '/'
# scriptdir = '/nsls2/data/srx/legacy/xf05id1/shared/src/bluesky_scripts/'

# Create the user directory
# try:
#     os.makedirs(userdatadir, exist_ok=True)
# except Exception as e:
#     print(e)
#     raise OSError('Cannot create directory: ' + userdatadir)

# Create the log file
# userlogfile = userdatadir + 'logfile' + str(saf_num) + '.txt'
# if not os.path.exists(userlogfile):
#     with open(userlogfile, 'w'):
#         ...

# Copy over the example scripts
# for f in ['simple_batch.py','fly_batch.py']:
#     if (not os.path.exists(userdatadir + f)):
#         shutil.copyfile(scriptdir + f, userdatadir + f)

# Update the symbolic link to the data
# try:
#     os.unlink('/nsls2/data/srx/legacy/xf05id1/shared/current_user_data')
# except FileNotFoundError:
#     print("[W] Previous user data directory link did not exist!")

# os.symlink(userdatadir, '/nsls2/data/srx/legacy/xf05id1/shared/current_user_data')

def get_stock_md(scan_md):
    scan_md['time_str'] =  ttime.ctime(ttime.time())
    if 'scan' not in scan_md:
        scan_md['scan'] = {}
    scan_md['scan']['energy'] = np.round(energy.energy.readback.get(), decimals=4)
    scan_md["beamline_id"] = "SRX"
    scan_md["md_version"] = "1.2"

    return scan_md


def get_det_md(scan_md, dets):

    # Starting from scratch
    if scan_md is None:
        scan_md = {}
    # Filling incomplete dictionary
    if (scan_md == {}
        or 'scan' not in scan_md):
        scan_md = get_stock_md(scan_md)
    
    # Create detectors dictionary. Overwrite existing.
    scan_md['scan']['detectors'] = {}
    
    # Add appropriate metadata for each detector
    for name, det in {d.name : d for d in dets}.items():
        if name == 'xs':
            det_dict = {}
            # Add the first four rois, regardless if they are used
            for ind in range(xs.channel01.get_mcaroi_count()):
                det_dict[f'roi{ind}'] = xs.channel01.get_mcaroi(mcaroi_number=ind + 1).roi_name.get()
        elif name == 'sclr1':
            det_dict = {}
        elif name == 'merlin':
            # Merlin attr is protected as operating_energy is non-standard
            det_dict = {}
            for obj, attr in [(det.cam, 'operating_energy')]:
                if hasattr(obj, attr):
                    det_dict[attr] = getattr(obj, attr).get()
        elif name == 'dexela':
            det_dict = {
                'binning_mode' : det.cam.binning_mode.get(as_string=True), # Detector binning
                'full_well_mode' : det.cam.full_well_mode.get(as_string=True), # Full well mode
            }
        elif name == 'nano_vlm':
            det_dict = {
                'cross_position_x' : det.over.overlay_1.position_x.get(), # Crosshair position x
                'cross_position_y' : det.over.overlay_1.position_y.get(), # Crosshair position y
            }
        elif name == 'xbpm2':
            det_dict = {}
        else:
            note_str = ('NOTE: Standard detector metadata has not yet be '
                        + f'implemented for {name}.')
            print(note_str)
            det_dict = {}
        
        # Add to scan_md
        scan_md['scan']['detectors'][name] = det_dict
    
    return scan_md


def logscan(scantype):
    h = db[-1]
    scan_id = h.start['scan_id']
    uid = h.start['uid']

    with open(userlogfile, 'a') as userlogf:
        userlogf.write(str(scan_id) + '\t' + uid + '\t' + scantype + '\n')


def logscan_event0info(scantype, event0info = []):
    h = db[-1]
    scan_id = h.start['scan_id']
    uid = h.start['uid']

    with open(userlogfile, 'a') as userlogf:
        userlogf.write(str(scan_id) + '\t' + uid + '\t' + scantype)
        events = list(db.get_events(h, stream_name='primary'))

        for item in event0info:
            userlogf.write('\t' + item + '=' + str(events[0]['data'][item]) + '\t')
        userlogf.write('\n')



def logscan_detailed(scantype):
    try:
        h = db[-1]
        scan_id = h.start['scan_id']
        uid = h.start['uid']

        with open(userlogfile, 'a') as userlogf:
            userlogf.write(str(scan_id) + '\t' + uid + '\t' + scantype + '\t' + str(h['start']['scan']['scan_input']) + '\n')

    except Exception:
        banner('Error writing to log file!')


def scantime(scanid, printresults=True):
    '''
    input: scanid
    return: start and stop time stamps as strings
    '''
    start_doc = db[scanid].start
    stop_doc = db[scanid].stop
    start_str = f"Scan start: {ttime.ctime(start_doc['time'])}"
    stop_str  = f"Scan stop : {ttime.ctime(stop_doc['time'])}"
    totaltime = stop_doc['time'] - start_doc['time']
    scannumpt = stop_doc['num_events']['primary']

    if (printresults):
        print(start_str)
        print(stop_str)
        print(f"Total time: {totaltime:.2f} s")
        print(f"Number of points: {scannumpt}")
        print(f"Scan time per point: {totaltime/scannumpt:.3f} s\n\n")
    return db[scanid].start['time'], db[scanid].stop['time'], start_str, stop_str


def timestamp_batchoutput(filename='timestamplog.text', initial_scanid=None, final_scanid=None):
    with open(filename, 'w') as f:
        for scanid in range(initial_scanid, final_scanid+1):
            f.write(str(scanid) + '\n')
            try:
                _, _, start_t, stop_t = scantime(scanid)
                f.write(start_t)
                f.write('\n')
                f.write(stop_t)
                f.write('\n')
            except:
                f.write('Scan did no finish correctly.\n')


def scantime_batchoutput(filename='scantimelog.txt', scanlist=[]):
    with open(filename, 'w') as f:
        f.write('scanid\tstartime(s)\tstoptime(s)\tstartime(date-time)\tstoptime(date-time)\n')
        for i in scanlist:
            starttime_s, endtime_s, starttime, endtime = scantime(i, printresults=False)
            f.write(str(i) + '\t' + str(starttime_s) + '\t' + str(endtime_s) + '\t' \
                    + starttime[12::] + '\t' + endtime[12::] + '\n')
