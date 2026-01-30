print(f'Loading {__file__}...')


import os
from ophyd import EpicsMotor, EpicsSignal
from ophyd import Device
from ophyd import Component as Cpt


# JJ Slits
class SRXJJSlits(Device):
    h_gap = Cpt(EpicsMotor, 'HA}Mtr')
    h_trans = Cpt(EpicsMotor, 'HT}Mtr')
    v_gap = Cpt(EpicsMotor, 'VA}Mtr')
    v_trans = Cpt(EpicsMotor, 'VT}Mtr')


jjslits = SRXJJSlits('XF:05IDD-OP:1{Slt:KB-Ax:', name='jjslits')


# Attenuator box
class SRXAttenuators(Device):
    Al_050um = Cpt(EpicsSignal, '8-Cmd')  # XF:05IDD-ES{IO:4}DO:8-Cmd
    Al_100um = Cpt(EpicsSignal, '7-Cmd')  # XF:05IDD-ES{IO:4}DO:7-Cmd
    Al_250um = Cpt(EpicsSignal, '6-Cmd')  # XF:05IDD-ES{IO:4}DO:6-Cmd
    Al_500um = Cpt(EpicsSignal, '5-Cmd')  # XF:05IDD-ES{IO:4}DO:5-Cmd
    Si_250um = Cpt(EpicsSignal, '2-Cmd')  # XF:05IDD-ES{IO:4}DO:2-Cmd
    Si_650um = Cpt(EpicsSignal, '1-Cmd')  # XF:05IDD-ES{IO:4}DO:1-Cmd


attenuators = SRXAttenuators('XF:05IDD-ES{IO:4}DO:', name='attenuators')


# PCOEdge detector motion
class SRXDownStreamGantry(Device):
    x = Cpt(EpicsMotor, 'X}Mtr')
    y = Cpt(EpicsMotor, 'Y}Mtr')
    z = Cpt(EpicsMotor, 'Z}Mtr')
    focus = Cpt(EpicsMotor, 'Foc}Mtr')


pcoedge_pos = SRXDownStreamGantry('XF:05IDD-ES:1{Det:3-Ax:',
                                  name='pcoedge_pos')
