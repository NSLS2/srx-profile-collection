# xrf map test
# To run this file:
# %run -i ~/.ipython/profile_collection/acceptance_tests/xrf_map_test.py

def xrf_map_test():
    print("Clearing suspenders")
    RE.clear_suspenders()
    print("Running xrf map scan with no shutters")
    uid, = RE(xrf_map(-10, 10, 51, -2, 2, 21, 0.1, sample_name='test', shutter=False))
    print("Scan complete.")
    print("Reading data from Tiled")
    data = c[uid]
    print(data)
    print("Successfully read data from Tiled")


xrf_map_test()
