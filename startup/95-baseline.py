# Setup baseline measurements for scans
print(f'Loading {__file__}...')

from bluesky.preprocessors import SupplementalData

sd = SupplementalData()

sd.baseline = []

baseline_components = [
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

# Only add connected signals.
# All signals should be full connected for normal beamline operations!
for component in baseline_components:
    if component.connected:
        sd.baseline.append(component)
    else:
        for sub_comp_name in component.component_names:
            sub_component = getattr(component, sub_comp_name)
            if sub_component.connected:
                sd.baseline.append(sub_component)
            else:
                warn_str = (f"WARNING: One or more PVs in ({sub_component.name}) are "
                            + "disconnected and not added to the baseline supplemental data!")
                print(warn_str)

RE.preprocessors.append(sd)

bec.disable_baseline()
