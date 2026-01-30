print(f'Loading {__file__}...')


import xraylib
import skbeam.core.constants.xrf as xrfC



def edge(element=None, line='K', unit='eV'):
    '''
    function return edge (K or L3) in eV or keV with input element sympbol
    '''

    atomic_num = xraylib.SymbolToAtomicNumber(element)

    if line == 'K':
        edge_value = xraylib.EdgeEnergy(atomic_num, xraylib.K_SHELL)
    if line == 'L3':
        edge_value = xraylib.EdgeEnergy(atomic_num, xraylib.L3_SHELL)        

    if unit == 'eV':
        return edge_value * 1000
    else:
        return edge_value


interestinglist = ['Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
                   'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As',
                   'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc',
                   'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
                   'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu',
                   'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                   'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
                   'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U']

elements = dict()
element_edges = ['ka1', 'ka2', 'kb1', 'la1', 'la2', 'lb1', 'lb2', 'lg1', 'ma1']
element_transitions = ['k', 'l1', 'l2', 'l3', 'm1', 'm2', 'm3', 'm4', 'm5']
for i in interestinglist:
    elements[i] = xrfC.XrfElement(i)


def setroi(roinum, element, edge=None, det=None):
    '''
    Set energy ROIs for Vortex SDD.
    Selects elemental edge given current energy if not provided.
    roinum      [1,2,3]     ROI number
    element     <symbol>    element symbol for target energy
    edge                    optional:  ['ka1', 'ka2', 'kb1', 'la1', 'la2',
                                        'lb1', 'lb2', 'lg1', 'ma1']
    '''
    cur_element = xrfC.XrfElement(element)
    if edge is None:
        for e in ['ka1', 'ka2', 'kb1', 'la1', 'la2',
                  'lb1', 'lb2', 'lg1', 'ma1']:
            if cur_element.emission_line[e] < energy.energy.readback.get():
                edge = 'e'
                break
    else:
        e = edge

    e_ch = int(cur_element.emission_line[e] * 1000)
    if det is not None:
        # we have been given an xspress3 that is not xs
        # look only at channel01
        # why?
        channels = [det.channels.channel01, ]
        #mcaroi = det.channels.channel01.get_mcaroi(
        #    mcaroi_number=roinum
        #)
        #mcaroi.configure_mcaroi(
        #    min_x=e_ch-100,
        #    size_x=200,
        #    roi_name=f"{element}_{e}"
        #)
        #mcaroi.kind = "hinted"
    else:
        # all channels on xs
        channels = list(xs.iterate_channels())

    for channel in channels:
        mcaroi = channel.get_mcaroi(
            mcaroi_number=roinum
        )
        # TODO: add eV-to-bin conversion to xspress3 class
        mcaroi.configure_mcaroi(
            min_x=(e_ch-100)/10,
            size_x=200/10,
            roi_name=f"{element}_{e}"
        )
        mcaroi.kind = "hinted"
    print("ROI{} set for {}-{} edge.".format(roinum, element, e))


def clearroi(roinum=None, verbose=False):
    if roinum is None:
        # leave roinum as None
        # it will be handled below so that all
        #   mcarois on each channel will be cleared
        pass
    else:
        try:
            roinum = list(roinum)
        except TypeError:
            roinum = [roinum]

    for channel in xs.iterate_channels():
        if roinum is None:
            # clear all mcarois for this channel
            roinums_to_clear = list(channel.mcaroi_numbers)
        else:
            # the user gave a single mcaroi number or a list
            #   of mcaroi numbers, we can just use roinum
            roinums_to_clear = roinum
        for mcaroi_number in roinums_to_clear:
            mcaroi = channel.get_mcaroi(mcaroi_number=mcaroi_number)
            mcaroi.clear()
            # TODO: add the following lines to mcaroi.clear()
            mcaroi.min_x.put(0)
            mcaroi.size_x.put(0)
            mcaroi.roi_name.put("")
            mcaroi.kind = "omitted"
            if verbose:
                print(f"cleared {mcaroi}")


def getemissionE(element, edge=None):
    cur_element = xrfC.XrfElement(element)
    if edge is None:
        print("Edge\tEnergy [keV]")
        for e in element_edges:
            if cur_element.emission_line[e] < 25. and \
               cur_element.emission_line[e] > 1.:
                # print("{0:s}\t{1:8.2f}".format(e, cur_element.emission_line[e]))
                print(f"{e}\t{cur_element.emission_line[e]:8.2f}")
    else:
        return np.round(cur_element.emission_line[edge], 3)


def getbindingE(element, edge=None):
    '''
    Return edge energy in eV if edge is specified,
    otherwise return K and L edge energies and yields
    element     <symbol>        element symbol for target
    edge        ['k','l1','l2','l3']    return binding energy of this edge
    '''
    if edge is None:
        y = [0., 'k']
        print("Edge\tEnergy [eV]\tYield")
        for i in ['k', 'l1', 'l2', 'l3']:
            print(f"{i}\t"
                  f"{xrfC.XrayLibWrap(elements[element].Z,'binding_e')[i]*1000.:8.2f}\t"
                  f"{xrfC.XrayLibWrap(elements[element].Z,'yield')[i]:5.3f}")
            if (y[0] < xrfC.XrayLibWrap(elements[element].Z, 'yield')[i] and
               xrfC.XrayLibWrap(elements[element].Z, 'binding_e')[i] < 25.):
                y[0] = xrfC.XrayLibWrap(elements[element].Z, 'yield')[i]
                y[1] = i
        return np.round(xrfC.XrayLibWrap(elements[element].Z, 'binding_e')[y[1]] * 1000., 3)
    else:
        return np.round(xrfC.XrayLibWrap(elements[element].Z, 'binding_e')[edge] * 1000., 3)
