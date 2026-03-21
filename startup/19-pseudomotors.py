print(f'Loading {__file__}...')


import numpy as np
from scipy.optimize import curve_fit

from ophyd import (
    EpicsSignal,
    EpicsSignalRO,
    EpicsMotor,
    Signal,
    PseudoPositioner,
    PseudoSingle,
)

from ophyd.pseudopos import pseudo_position_argument, real_position_argument
from ophyd import Component as Cpt


class ProjectedTopStage(PseudoPositioner):

    # Pseudo axes
    projx = Cpt(PseudoSingle)
    projz = Cpt(PseudoSingle)

    # Real axes. From SRXNanoStage class definition.
    topx = Cpt(EpicsMotor, 'XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr.RBV
    topz = Cpt(EpicsMotor, 'XF:05IDD-ES:1{nKB:Smpl-Ax:zth}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:zth}Mtr.RBV

    # Configuration signals
    th = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:th}Mtr.RBV')  # XF:05IDD-ES:1{nKB:Smpl-Ax:th}Mtr.RBV
    velocity_x = Cpt(EpicsSignal, 'XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr.VELO')
    velocity_z = Cpt(EpicsSignal, 'XF:05IDD-ES:1{nKB:Smpl-Ax:zth}Mtr.VELO')
    acceleration_x = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr.ACCL')
    acceleration_z = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:zth}Mtr.ACCL')
    motor_egu_x = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr.EGU')
    motor_egu_z = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:zth}Mtr.EGU')

    # Dumb way to overwrite the hard-coded Signal class limits
    class LimitedSignal(Signal):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._signal_limits = (0, 0)
        
        @property
        def limits(self):
            return self._signal_limits
        
        # Not as setter to help avoid non-explicit calls
        def _set_signal_limits(self, new_limits):
            new_limits = tuple(new_limits)
            if len(new_limits) != 2:
                err_str = ('Length of new limits must be 2, not '
                           + f'{len(new_limits)}.')
                raise ValueError(err_str)

            self._signal_limits = new_limits


    # Create projected signals to read
    velocity = Cpt(LimitedSignal, None, add_prefix=(), kind='config')
    acceleration = Cpt(LimitedSignal, None, add_prefix=(), kind='config')
    motor_egu = Cpt(LimitedSignal, None, add_prefix=(), kind='config')

    def __init__(self,
                 *args,
                 projected_axis=None,
                 **kwargs):

        super().__init__(*args, **kwargs)
        
        # Store projected axis for determining projected velocity
        if projected_axis is None:
            err_str = "Must define projected_axis as 'x' or 'z'."
            raise ValueError(err_str)
        elif str(projected_axis).lower() not in ['x', 'z']:
            err_str = ("ProjectedTopStage axis only supported for 'x' "
                       + f"or 'z' projected axis not {projected_axis}.")
            raise ValueError(err_str)
        self._projected_axis = str(projected_axis).lower()

        # Define defualt projected signals
        velocity = min([self.velocity_x.get(),
                        self.velocity_z.get()])
        acceleration = min([self.acceleration_x.get(),
                            self.acceleration_z.get()])
        if self.motor_egu_x.get() == self.motor_egu_z.get():
            motor_egu = self.motor_egu_x.get()
        else:
            err_str = (f'topx motor_egu of {self.motor_egu_x.get()} does '
                       + 'not match topz motor_egu of '
                       + f'{self.motor_egu_z.get()}')
            raise AttributeError(err_str)

        self.velocity.set(velocity)
        self.acceleration.set(acceleration)
        self.motor_egu.set(motor_egu)
        self.motor_egu._set_signal_limits((None, None))

        # Set velocity limits
        velocity_limits = (
            max([self.velocity_x.low_limit,
                 self.velocity_z.low_limit]),
            min([self.velocity_x.high_limit,
                 self.velocity_z.high_limit])
        )
        self.velocity._set_signal_limits(velocity_limits)

        # Set acceleration limits
        acceleration_limits = (
            max([self.acceleration_x.low_limit,
                 self.acceleration_z.low_limit]),
            min([self.acceleration_x.high_limit,
                 self.acceleration_z.high_limit])
        )
        self.acceleration._set_signal_limits(acceleration_limits)

        # Set up alias for flyer readback
        if self._projected_axis == 'x':
            self.user_readback = self.projx.readback
        else:
            self.user_readback = self.projz.readback

    # Convenience function to get rotation matrix between 
    # rotated top stage axes and projected lab axes
    def R(self):
        th = self.th.get()
        th = np.radians(th / 1000) # to radians
        return np.array([[np.cos(th), np.sin(th)],
                         [-np.sin(th), np.cos(th)]])
    

    # Function to change component motor velocities
    def set_component_velocities(self,
                                 topx_velocity=None,
                                 topz_velocity=None):
        
        bool_flags = sum([topx_velocity is None,
                          topz_velocity is None])

        if bool_flags == 1:
            err_str = ('Must specify both topx_velocity and '
                       + 'topz_velocity or neither.')
            raise ValueError(err_str)
        elif bool_flags == 2:
            # Determine component velocities from projected
            velocity = self.velocity.get()
            if self._projected_axis == 'x':
                velocity_vector = [velocity, 0]
            else:
                velocity_vector = [0, velocity]

            (topx_velocity,
             topz_velocity) = np.abs(self.R() @ velocity_vector)
        
        if topx_velocity < self.topx.velocity.low_limit:
            topx_velocity = self.topx.velocity.low_limit
        if topz_velocity < self.topz.velocity.low_limit:
            topz_velocity = self.topz.velocity.low_limit
        
        # In the background is a set_and_wait. Returning status object may not be necessary
        self.velocity_x.set(topx_velocity)
        # print(f'{topx_velocity=}')
        self.velocity_z.set(topz_velocity)
        # print(f'{topz_velocity=}')
        # print('finished changing velocities')

    
    # Wrap move function with stage_sigs-like behavior
    def move(self, *args, timeout=60, **kwargs):

        # Get starting velocities
        start_topx_velocity = self.velocity_x.get()
        start_topz_velocity = self.velocity_z.get()
        
        # Set component velocities based on internal velocity signal
        # print('setting velocities')
        self.set_component_velocities()

        # Move like normal
        # print('starting move')
        mv_st = super().move(*args, **kwargs)
        mv_st.wait(timeout=timeout)
        # print('move done')

        # Reset component velocities to original values
        # print('resetting velocities')
        self.set_component_velocities(
                    topx_velocity=start_topx_velocity,
                    topz_velocity=start_topz_velocity)
        
        # Must return move status object!!
        return mv_st


    def _forward(self, projx, projz):
        #     # |topx|   | cos(th)  sin(th)| |projx|
        #     # |topz| = |-sin(th)  cos(th)| |projz|
        return self.R().T @ [projx, projz]

    
    def _inverse(self, topx, topz):
        #     # |projx|   |cos(th)  -sin(th)| |topx|
        #     # |projz| = |sin(th)   cos(th)| |topz|
        return self.R() @ [topx, topz]


    @pseudo_position_argument
    def forward(self, p_pos):

        if self._projected_axis == 'x':
            projx = p_pos.projx
            self.projz.sync() # Ignore setpoint value
            projz = p_pos.projz
        else:
            projz = p_pos.projz
            self.projx.sync()
            projx = p_pos.projx
        
        topx, topz = self._forward(projx, projz)
        return self.RealPosition(topx=topx, topz=topz)


    @real_position_argument
    def inverse(self, r_pos):
        topx = r_pos.topx
        topz = r_pos.topz
        projx, projz = self._inverse(topx, topz)
        return self.PseudoPosition(projx=projx, projz=projz)


# Enable these once we have topx and topz again
# projx = ProjectedTopStage(name='projected_top_x', projected_axis='x')
# projz = ProjectedTopStage(name='projected_top_z', projected_axis='z')


# def projected_scan_and_fly(*args, extra_dets=None, center=True, **kwargs):
#     kwargs.setdefault('xmotor', projx)
#     kwargs.setdefault('ymotor', nano_stage.y)
#     kwargs.setdefault('flying_zebra', nano_flying_zebra_coarse)
#     yield from abs_set(kwargs['flying_zebra'].fast_axis, 'NANOHOR')
#     yield from abs_set(kwargs['flying_zebra'].slow_axis, 'NANOVER')

#     _xs = kwargs.pop('xs', xs)
#     if extra_dets is None:
#         extra_dets = []
#     dets = [_xs] + extra_dets

#     if center:
#         yield from move_to_map_center(*args, **kwargs)
#     yield from scan_and_fly_base(dets, *args, **kwargs)
#     if center:
#         yield from move_to_map_center(*args, **kwargs)


# def move_to_map_center(*args, **kwargs):
#     xmotor = kwargs['xmotor']
#     ymotor = kwargs['ymotor']

#     xstart, xend, xnum, ystart, yend, ynum, dwell = args

#     xcen = xstart + ((xend - xstart) / 2)
#     ycen = ystart + ((yend - ystart) / 2)

#     # print(f'Move to {xcen} xcen')
#     # print(f'Move to {ycen} ycen.')
#     yield from mv(xmotor, xcen,
#                   ymotor, ycen)


# Not needed with working top stages
class CompucentricRotation(PseudoPositioner):

    # Pseudo axes
    rotation = Cpt(PseudoSingle)

    # Real axes
    # Rotation
    th = Cpt(EpicsMotor, 'XF:05IDD-ES:1{nKB:Smpl-Ax:th}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:th}Mtr.RBV
    # Coarse motors
    x = Cpt(EpicsMotor, 'XF:05IDD-ES:1{nKB:Smpl-Ax:sx}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:sx}Mtr.RBV
    z = Cpt(EpicsMotor, 'XF:05IDD-ES:1{nKB:Smpl-Ax:sz}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:sz}Mtr.RBV
    # Fine motors
    # sx = Cpt(EpicsMotor, 'XF:05IDD-ES:1{nKB:Smpl-Ax:ssx}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:ssx}Mtr.RBV
    # sz = Cpt(EpicsMotor, 'XF:05IDD-ES:1{nKB:Smpl-Ax:ssz}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:ssz}Mtr.RBV


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.r = 0
        self.th0 = 0
        self._mode = 'coarse'

    
    @property
    def mode(self):
        return self._mode

    
    def calculate_deltas(self, new_th):
        # new_th in mdeg

        curr_th = self.real_position.th # in mdeg
        curr_phi = np.radians((curr_th + self.th0) / 1e3)
        new_phi = np.radians((new_th + self.th0) / 1e3)
        # phi in radians

        # Negative dx for some reason
        dx = -self.r * (np.cos(new_phi) - np.cos(curr_phi))
        dz = self.r * (np.sin(new_phi) - np.sin(curr_phi))

        return dx, dz


    def set_mode(self, key='fine'):
        if key.lower() == 'fine':
            self._mode = 'fine'
        elif key.lower() == 'coarse':
            self._mode = 'coarse'
        else:
            raise ValueError(f"Unknown motor key {key}. Only 'fine' and 'coarse' accepted.")
        
    
    @pseudo_position_argument
    def forward(self, new_th):

        dx, dz = self.calculate_deltas(new_th.rotation)
        return self.RealPosition(x=self.real_position.x + dx,
                                 z=self.real_position.z + dz,
                                 th=new_th.rotation)


    @real_position_argument
    def inverse(self, real_pos):
        return self.PseudoPosition(rotation=real_pos.th)
    


        # if self.mode == 'coarse':
        #     return self.RealPosition(x=self.real_position.x + dx,
        #                              z=self.real_position.z + dz,
        #                              th=new_th)
        # elif self.mode == 'fine':
        #     return self.RealPosition(x=self.real_position.sx + dx,
        #                              z=self.real_position.sz + dz,
        #                              th=new_th)


# Should behave just like th until the calibration function has been ran
comp_th = CompucentricRotation(name='compucentric_th')


# Compucentric fitting math lives here in case the measurements are taken manually

# Functional form of x offsets from perfectly cirucular motion
def compucentric_x_model(th, r, th0, x0):
    # Negative for some unkown reason???
    return -r * np.cos(th + th0) + x0


# Functional form of x offsets from perfectly cirucular motion
def compucentric_z_model(th, r, th0, x0):
    return r * np.sin(th + th0) + x0


# Combined offsets
def compucentric_combined_model(th, r, th0, x0, z0):
    x = compucentric_x_model(th, r, th0, x0)
    z = compucentric_z_model(th, r, th0, z0)
    return np.concatenate([x, z])


def fit_compucentric_model(th, x_obs=None, z_obs=None,
                           correct_pseudomotor=True,
                           plot=True):

    bounds = [[0, -np.pi, -np.inf],
              [np.inf, np.pi, np.inf]]
    observed = None
    labels = []
    if x_obs is not None:
        observed = 'x'
        labels.append('x0')
        if len(th) != len(x_obs):
            raise ValueError('All observations must have same length.')
        obs = x_obs
        model = compucentric_x_model
        p0 = [(np.max(x_obs) - np.min(x_obs)) / 2,
              0, # overwritten
              np.mean(x_obs)]
    if z_obs is not None:
        observed = 'z'
        labels.append('z0')
        if len(th) != len(z_obs):
            raise ValueError('All observations must have same length.')
        obs = z_obs
        model = compucentric_z_model
        p0 = [(np.max(z_obs) - np.min(z_obs)) / 2,
              0, # overwritten
              np.mean(z_obs)]
    if x_obs is not None and z_obs is not None:
        observed = 'both'
        obs = np.concatenate([x_obs, z_obs])
        model = compucentric_combined_model
        p0 = [np.mean([(np.max(x_obs) - np.min(x_obs)) / 2,
                       (np.max(z_obs) - np.min(z_obs)) / 2]),
              0,
              np.mean(x_obs), np.mean(z_obs)
              ]
        bounds[0].append(-np.inf)
        bounds[1].append(np.inf)
    if observed is None:
        raise ValueError('Must define one or both of x_obs or z_obs.')

    succ, fits, errs, r_sqr = [], [], [], []
    for rad in np.radians([-180, 0, 180]):
        p0[1] = rad

        try:
            popt, pcov = curve_fit(model, th, obs, p0=p0, bounds=bounds)
            succ.append(True)
        except Exception as e:
            succ.append(False)
            err = e
            popt = p0
            pcov = np.eye(len(p0)) * 1e10
        
        perr = np.sqrt(np.diag(pcov))
        fits.append(popt)
        errs.append(perr)
        
        # Calculate r_squared
        obs_pred = model(th, *popt)
        res = obs - obs_pred
        ss_res = np.sum(res**2)
        ss_tot = np.sum((obs - np.mean(obs))**2)
        if ss_tot != 0:
            r_squared = 1 - (ss_res / ss_tot)
        else:
            r_squared = 0
        r_sqr.append(r_squared)

    if not any(succ):
        # raise ValueError('All fitting failed.')
        print(f'All Fitting Failed: {err}')

    # print([p[0] for p in fits])
    # print([np.degrees(p[1]) for p in fits])
    # print(r_sqr)

    best_idx = np.argmax(r_sqr)
    r_fit, th0_fit = fits[best_idx][:2]
    r_err, th0_err = errs[best_idx][:2]
    r_rel_err = (r_err / r_fit) * 100 if r_fit != 0 else np.inf

    print('Fit Results:')
    print(f'R-squred is {r_sqr[best_idx]:.7f}')
    print(f'\tradius = {r_fit:.3f} ± {r_err:.3f} μm')
    print(f'\tth0 = {np.degrees(th0_fit):.3f} ± {np.degrees(th0_err):.3f} deg')

    for i, (off_fit, off_err) in enumerate(zip(fits[best_idx][2:],
                                               errs[best_idx][2:])):
        print(f'\t{labels[i]} = {off_fit:.3f} ± {off_err:.3f} μm')

    if plot:
        fit_th = np.linspace(np.min(th), np.max(th), 5 * len(th))
        if observed == 'both':
            x_pred, z_pred = np.split(model(fit_th, *fits[best_idx]), 2)
        elif observed == 'x':
            x_pred = model(fit_th, *fits[best_idx])
        elif observed == 'z':
            z_pred = model(fit_th, *fits[best_idx])

        if x_obs is not None:
            fig, ax = plt.subplots()
            ax.scatter(np.degrees(th), x_obs, c='r', label='Measured')
            ax.plot(np.degrees(fit_th), x_pred, c='k', label='Fit')
            ax.set_xlabel('Angle [deg]')
            ax.set_ylabel('X Position [μm]')
            ax.set_title('Compucentric X position Fit')
            ax.legend()
            fig.show()
        if z_obs is not None:
            fig, ax = plt.subplots()
            ax.scatter(np.degrees(th), z_obs, c='r', label='Measured')
            ax.plot(np.degrees(fit_th), z_pred, c='k', label='Fit')
            ax.set_xlabel('Angle [deg]')
            ax.set_ylabel('Z Position [μm]')
            ax.set_title('Compucentric Z position Fit')
            ax.legend()
            fig.show()


    if r_rel_err > 10:
        print(('WARNING: Poor calibration. Relative error of radius greater than 10%.'
                + '\nLarger angular range may improve fitting.'))

    if correct_pseudomotor:
        comp_th.r = r_fit
        comp_th.th0 = np.degrees(th0_fit) * 1e3
    else:
        return fits[best_idx]
