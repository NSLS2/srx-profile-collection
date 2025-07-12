print(f'Loading {__file__}...')


from ophyd import (Device, EpicsMotor, EpicsSignal, EpicsSignalRO,
                   PVPositionerPC)
from ophyd import Component as Cpt


# nano-KB mirrors
class SRXNanoKBCoarse(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, 'SPOS')  # XF:05IDD-ES:1{Mir:nKBv-Ax:PC}SPOS
    readback = Cpt(EpicsSignalRO, 'RPOS')  # XF:05IDD-ES:1{Mir:nKBh-Ax:PC}RPOS


class SRXNanoKB(Device):
    # XF:05IDD-ES:1{nKB:vert-Ax:Y}Mtr.RBV
    v_y = Cpt(EpicsMotor, 'vert-Ax:Y}Mtr')
    # XF:05IDD-ES:1{nKB:vert-Ax:PC}RPOS
    v_pitch = Cpt(SRXNanoKBCoarse,
                  'XF:05IDD-ES:1{nKB:vert-Ax:PC}',
                  name='nanoKB_v_pitch',
                  add_prefix=())  
    # XF:05IDD-ES:1{nKB:horz-Ax:PC}Mtr.RBV
    v_pitch_um = Cpt(EpicsMotor, 'vert-Ax:PC}Mtr')
    # XF:05IDD-ES:1{nKB:vert-Ax:PFPI}Mtr.RBV
    v_pitch_fine = Cpt(EpicsMotor,
                       "XF:05IDD-ES:1{nKB:vert-Ax:PFPI}Mtr",
                       name="v_pitch_fine",
                       add_prefix="")
    # XF:05IDD-ES:1{nKB:horz-Ax:X}Mtr.RBV
    h_x = Cpt(EpicsMotor, 'horz-Ax:X}Mtr')
    # XF:05IDD-ES:1{nKB:horz-Ax:PC}RPOS
    h_pitch = Cpt(SRXNanoKBCoarse,
                  'XF:05IDD-ES:1{nKB:horz-Ax:PC}',
                  name='nanoKB_h_pitch',
                  add_prefix=())
    # XF:05IDD-ES:1{nKB:vert-Ax:PC}Mtr.RBV
    h_pitch_um = Cpt(EpicsMotor, 'horz-Ax:PC}Mtr')
    # XF:05IDD-ES:1{nKB:horz-Ax:PFPI}Mtr.RBV
    h_pitch_fine = Cpt(EpicsMotor,
                       "XF:05IDD-ES:1{nKB:horz-Ax:PFPI}Mtr",
                       name="h_pitch_fine",
                       add_prefix="")


nanoKB = SRXNanoKB('XF:05IDD-ES:1{nKB:', name='nanoKB')


# High flux sample stages
class SRXNanoStage(Device):
    # x = Cpt(EpicsMotor, 'sx}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:sx}Mtr.RBV
    y = Cpt(EpicsMotor, 'sy}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:sy}Mtr.RBV
    z = Cpt(EpicsMotor, 'sz}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:sz}Mtr.RBV
    sx = Cpt(EpicsMotor, 'ssx}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:ssx}Mtr.RBV
    sy = Cpt(EpicsMotor, 'ssy}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:ssy}Mtr.RBV
    sz = Cpt(EpicsMotor, 'ssz}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:ssz}Mtr.RBV
    th = Cpt(EpicsMotor, 'th}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:th}Mtr.RBV
    topx = Cpt(EpicsMotor, 'xth}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr.RBV
    # topz = Cpt(EpicsMotor, 'zth}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:zth}Mtr.RBV


nano_stage = SRXNanoStage('XF:05IDD-ES:1{nKB:Smpl-Ax:', name='nano_stage')


# nanoVLM motion
class SRXNanoVLMStage(Device):
    x = Cpt(EpicsMotor, 'X}Mtr')  # XF:05IDD-ES:1{nKB:VLM-Ax:X}Mtr.RBV
    y = Cpt(EpicsMotor, 'Y}Mtr')
    z = Cpt(EpicsMotor, 'Z}Mtr')


nano_vlm_stage = SRXNanoVLMStage('XF:05IDD-ES:1{nKB:VLM-Ax:',
                                 name='nano_vlm_stage')


# SDD motion
class SRXNanoDet(Device):
    x = Cpt(EpicsMotor, 'X}Mtr')  # XF:05IDD-ES:1{nKB:Det-Ax:X}Mtr.RBV
    y = Cpt(EpicsMotor, 'Y}Mtr')  # XF:05IDD-ES:1{nKB:Det-Ax:Y}Mtr.RBV
    z = Cpt(EpicsMotor, 'Z}Mtr')  # XF:05IDD-ES:1{nKB:Det-Ax:Z}Mtr.RBV
    sample2detector = Cpt(EpicsSignalRO,
                          "XF:05IDD-ES:1{nKB:Det-Ax:Dist}MTR:RBV",
                          name="sample2detector",
                          add_prefix="")


nano_det = SRXNanoDet('XF:05IDD-ES:1{nKB:Det-Ax:', name='nano_det')


# Lakeshore temperature monitors
class SRXNanoTemp(Device):
    temp_nanoKB_horz = Cpt(EpicsSignalRO, '2}T:C-I')
    temp_nanoKB_vert = Cpt(EpicsSignalRO, '1}T:C-I')
    temp_nanoKB_base = Cpt(EpicsSignalRO, '4}T:C-I')
    temp_room = Cpt(EpicsSignalRO, '3}T:C-I')


temp_nanoKB = SRXNanoTemp('XF:05IDD-ES{LS:1-Chan:', name='temp_nanoKB')

# NanoKB inteferometer monitors
class SRXNanoKBInterferometer(Device):
    posX = Cpt(EpicsSignalRO, 'Chan0}Pos-I')
    posY = Cpt(EpicsSignalRO, 'Chan1}Pos-I')
    posZ = Cpt(EpicsSignalRO, 'Chan2}Pos-I')


nanoKB_interferometer = SRXNanoKBInterferometer('XF:05IDD-ES:1{FPS:1-', name='nanoKB_interferometer')


# Nanostage inteferometer monitors
class SRXNanoStageInterferometer(Device):
    posX = Cpt(EpicsSignalRO, 'POS_0')
    posY = Cpt(EpicsSignalRO, 'POS_1')
    posZ = Cpt(EpicsSignalRO, 'POS_2')


nano_stage_interferometer = SRXNanoStageInterferometer('XF:05IDD-ES:1{PICOSCALE:1}', name='nano_stage_interferometer')


def move_to_scanner_center(timeout=10):
    """
    Move the scanner stages to the center (zero).

    Parameters
    ----------
    timeout : integer
        Amount of time in seconds to wait

    Yields
    ------
    msg : Msg to move motors to zero
    """
    return (yield from mv(nano_stage.sx, 0,
                          nano_stage.sy, 0,
                          nano_stage.sz, 0,
                          timeout=timeout))


def reset_scanner_velocity():
    """
    Reset the scanner stages to their nominal speeds
    """
    for d in [nano_stage.topx, nano_stage.y, nano_stage.z]:
        d.velocity.set(500)  # um/s
    for d in [nano_stage.sx, nano_stage.sy, nano_stage.sz]:
        d.velocity.set(100)  # um/s
    nano_stage.th.velocity.set(10_000)  # mdeg/s


def center_scanner():
    """
    Move the coarse and scanner stages such that the scanner is set to
    zero and the coarse stages translate to keep the sample in position
    """
    del_sx = nano_stage.sx.user_readback.get()
    del_sy = nano_stage.sy.user_readback.get()
    del_sz = nano_stage.sz.user_readback.get()

    yield from mv(nano_stage.sx, 0)
    yield from mvr(nano_stage.topx, del_sx)

    yield from mv(nano_stage.sy, 0)
    yield from mvr(nano_stage.y, del_sy)

    yield from mv(nano_stage.sz, 0)
    yield from mvr(nano_stage.z, del_sz)


def mv_along_axis(z_end):
    """
    Move the sample along the focused beam axis
    """
    cur_x = nano_stage.topx.user_readback.get()
    cur_y = nano_stage.y.user_readback.get()
    cur_z = nano_stage.z.user_readback.get()
    print(f'Current locations are: {cur_x}, {cur_y}, {cur_z}')

    ratio_xz = 0.004875
    ratio_yz = 0.0067874

    delta_z = z_end - cur_z
    print(f'Move z to {z_end}')

    delta_x = ratio_xz * delta_z
    print(f'Move x by {delta_x}')

    delta_y = ratio_yz * delta_z
    print(f'Move y by {delta_y}')

    yield from mvr(nano_stage.topx, delta_x)
    yield from mvr(nano_stage.y, delta_y)
    yield from mv(nano_stage.z, z_end)

