# Deployment acceptance tests
# Running the tests from IPython
# %run -i ~/.ipython/profile_collection/acceptance_tests/test_deployments.py

import os

def test_set_roi():
    """
    Set ROI, and check PV values to see that it has been set correctly.
    """
    print("Starting test_set_roi")
    setroi(1, 'Fe')
    input("Check CSS, and press any key to continue")
    print("setroi complete")
    #assert xs.channel1.rois.roi01.bin_high.get() == 650
    #assert xs.channel1.rois.roi01.bin_low.get() == 630


def test_clear_roi():
    print("Starting test_crear_roi")
    clearroi()
    input("Check CSS, and press any key to continue")
    print("Test clear_roi complete, check CSS")


def test_get_binding_energies():
    print("Starting test_get_binding_energies")
    result = getbindingE('Fe')
    assert result == 7112.0
    print("test_get_binding_energies complete")


def test_get_emmision_energies():
    print("Starting test_get_emmision_energies")
    getemissionE('Fe')
    print("test_get_emmision_energies complete")


def test_fly_scan1():
    """
    Fly scan test 1.
    If db.table() and export scan complete without errors than it was successful.
    """
    print("Starting fly scan test 2")
    RE.clear_suspenders()
    RE.preprocessors.clear()
    uid, = RE(nano_scan_and_fly(-5, 5, 11, 0, 2, 3, 0.1, shutter=False))
    print("Fly scan complete")
    flush_tiled()  # wait for buffered documents to reach Tiled
    print("Reading scan from Tiled: 'primary' stream")
    c[uid]["primary"]["data"].read()
    print("Reading scan from Tiled: 'stream0' stream")
    c[uid]["stream0"]["data"].read()
    print("Loading data using PyXRF")
    from pyxrf.api import make_hdf
    dir_name = "/tmp/acceptance_tests"
    os.makedirs(dir_name, exist_ok=True)
    make_hdf(uid, wd=dir_name, create_each_det=True, file_overwrite_existing=True)
    print("Test is complete")


def test_fly_scan2():
    """
    Fly scan test 2.
    If db.table() and export scan complete without errors than it was successful.
    """
    print("Starting fly scan test 2")
    RE.clear_suspenders()
    RE.preprocessors.clear()
    uid, = RE(nano_y_scan_and_fly(-5, 5, 11, 0, 2, 3, 0.1, shutter=False))
    print("Fly scan complete")
    flush_tiled()  # wait for buffered documents to reach Tiled
    print("Reading scan from Tiled: 'primary' stream")
    c[uid]["primary"]["data"].read()
    print("Reading scan from Tiled: 'stream0' stream")
    c[uid]["stream0"]["data"].read()
    print("Loading data using PyXRF")
    from pyxrf.api import make_hdf
    dir_name = "/tmp/acceptance_tests"
    os.makedirs(dir_name, exist_ok=True)
    make_hdf(uid, wd=dir_name, create_each_det=True, file_overwrite_existing=True)
    print("Test is complete")


def test_step_scan():
    """
    Step scan test.
    If db.table() and export scan complete without errors than it was successful.
    """
    print("Starting fly scan test 2")
    uid, = RE(nano_xrf(-5, 5, 1, 0, 2, 1, 0.1, shutter=False))
    print("Fly scan complete")
    flush_tiled()  # wait for buffered documents to reach Tiled
    print("Reading scan from Tiled: 'primary' stream")
    c[uid]["primary"]["data"].read()
    print("Reading scan from Tiled: 'stream0' stream")
    c[uid]["stream0"]["data"].read()
    print("Loading data using PyXRF")
    from pyxrf.api import make_hdf
    dir_name = "/tmp/acceptance_tests"
    os.makedirs(dir_name, exist_ok=True)
    make_hdf(uid, wd=dir_name, create_each_det=True, file_overwrite_existing=True)
    print("Test is complete")


# Manual tests
# - XANES data appears on a network shared mount
# - Data is being copied from the IOC to the GPFS
# - Athena starts
# - Athena can access and open the XANES data that was just collected
# - Data can be downloaded from Data Broker
# - run-pyxrftools
# - ] make_hdf(12345)
# - PyXRF starts
# - run-pyxrf
# - PyXRF can open the imaging data that was just collected

test_set_roi()
test_clear_roi()
test_get_binding_energies()
test_get_emmision_energies()
test_fly_scan1()
test_fly_scan2()
test_step_scan()
