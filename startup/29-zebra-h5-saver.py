# from ophyd import EpicsSignal, EpicsSignalRO, Device, Component as Cpt


# class ZebraSaver(Device):
#     # Saving business logic:
#     write_dir = Cpt(EpicsSignal, "write_dir", string=True)
#     file_name = Cpt(EpicsSignal, "file_name", string=True)
#     full_file_path = Cpt(EpicsSignalRO, "full_file_path")

#     acquire = Cpt(EpicsSignal, "acquire", string=True)
#     file_stage = Cpt(EpicsSignal, "stage")


#     dev_type = Cpt(EpicsSignal, "dev_type")

#     # Zebra-related PVs:
#     enc1 = Cpt(EpicsSignal, "enc1")
#     enc2 = Cpt(EpicsSignal, "enc2")
#     enc3 = Cpt(EpicsSignal, "enc3")
#     zebra_time = Cpt(EpicsSignal, "zebra_time")

#     # Scaler-related PVs:
#     i0 = Cpt(EpicsSignal, "i0")
#     im = Cpt(EpicsSignal, "im")
#     it = Cpt(EpicsSignal, "it")
#     sis_time = Cpt(EpicsSignal, "sis_time")

# class ZebraSaver(Device):
#     # Saving business logic:
#     write_dir = Cpt(EpicsSignal, "write_dir", string=True)
#     file_name = Cpt(EpicsSignal, "file_name", string=True)
#     full_file_path = Cpt(EpicsSignalRO, "full_file_path")

#     acquire = Cpt(EpicsSignal, "acquire", string=True)
#     file_stage = Cpt(EpicsSignal, "stage")


#     dev_type = Cpt(EpicsSignal, "dev_type")

#     # Zebra-related PVs:
#     enc1 = Cpt(EpicsSignal, "enc1")
#     enc2 = Cpt(EpicsSignal, "enc2")
#     enc3 = Cpt(EpicsSignal, "enc3")
#     zebra_time = Cpt(EpicsSignal, "zebra_time")

#     # Scaler-related PVs:
#     for component_name in sclr1.scaler_name_list:
#         setattr(component_name, Cpt(EpicsSignal, component_name))
#     sis_time = Cpt(EpicsSignal, "sis_time")


# zs = ZebraSaver("XF:05IDD-ES{ZebraSaver:1}:", name="zs")
