# Setup baseline measurements for scans
print(f'Loading {__file__}...')

from bluesky.preprocessors import SupplementalData

sd = SupplementalData()

sd.baseline = [
               ring_current,                                      # Storage ring current
               fe, energy, dcm, hfm,                              # Front-end slits, Undulator/Bragg, HDCM, HFM
               slt_wb, slt_pb, slt_ssa,                           # White-, Pink-Beam slits, SSA
               jjslits, attenuators,                              # JJ slits, Attenuator Box
               nanoKB, nano_vlm_stage, nano_det, temp_nanoKB,     # nanoKBs, VLM, Detector, Temperatures
               nano_stage,                                        # coarse/fine sample stages
               nanoKB_interferometer, nano_stage_interferometer,  # nanoKB interferometer, sample interferometer
               xs.cam.ctrl_dtc,                                   # X3X DTC enabled
               i0_preamp, im_preamp, it_preamp,
               ]

RE.preprocessors.append(sd)

bec.disable_baseline()
