print(f'Loading {__file__}...')


import time as ttime
from bluesky.callbacks import CallbackBase,LivePlot
from xray_vision.backend.mpl.cross_section_2d import CrossSection

# Try-except only hear for testing. Can be removed after validation.
_start_doc_cache = {}


def cb_print_scaninfo(name, doc):
    if name == "start":
        # Keep only the most recent start document (used by the stop banner)
        _start_doc_cache.clear()
        _start_doc_cache[doc['uid']] = doc
        try:
            str1 = f"Scan Type:  {doc['scan']['type']}"
        except KeyError:
            str1 = f"Scan Type:  {doc['plan_name'].upper()}"
        except:
            str1 = ''
        str2 = f"Scan ID:    {doc['scan_id']}"
        str3 = f"Start Time: {ttime.ctime(doc['time'])}"
        banner([str1, str2, str3])
    elif name == "stop":
        # Look the start document up in the local cache: the legacy Mongo
        # broker is disconnected and Tiled lags behind Bluesky (documents
        # are buffered).
        start_doc = _start_doc_cache.get(doc['run_start'])

        if start_doc is not None:
            try:
                str1 = f"Scan Type:    {start_doc['scan']['type']}"
            except KeyError:
                str1 = f"Scan Type:  {start_doc['plan_name'].upper()}"
            except:
                str1 = ''
            str2 = f"Scan ID:      {start_doc['scan_id']}"
            str3 = f"Stop Time:    {ttime.ctime(doc['time'])}"
            dt = doc['time'] - start_doc['time']
            if dt < 60:
                str4 = f"  Total Time: {dt:.2f} s"
            elif dt < 3600:
                str4 = f"  Total Time: {dt/60:.2f} min"
            else:
                str4 = f"  Total Time: {dt/3600:.2f} h"
            banner([str1, str2, str3, str4])
        else: 
            str3 = f"Stop Time:    {ttime.ctime(doc['time'])}"
            banner([str3])


RE.subscribe(cb_print_scaninfo)


i0_baseline = 7.24e-10


# LivePlot for Flying XANES measurements
@make_class_safe(logger=logger)
class LivePlotFlyingXAS(QtAwareCallback):
    def __init__(self, y, y_norm=None, x=None, e_pts=None, xlabel=None, *, legend_keys=None, xlim=None, ylim=None,
                 ax=None, fig=None, epoch='run', **kwargs):
        super().__init__(use_teleporter=kwargs.pop('use_teleporter', None))
        self.__setup_lock = threading.Lock()
        self.__setup_event = threading.Event()

        self._xind = 0
        self._xlabel = xlabel
        self._y_norm = y_norm
        self._y_norm_val = 1
        print(f'{self._y_norm_val=}')
        if e_pts is None:
            raise RuntimeError("Energy points are required")
        self._epts = e_pts


        def setup():
            # Run this code in start() so that it runs on the correct thread.
            nonlocal y, x, legend_keys, xlim, ylim, ax, fig, epoch, kwargs
            import matplotlib.pyplot as plt
            with self.__setup_lock:
                if self.__setup_event.is_set():
                    return
                self.__setup_event.set()
            if fig is not None:
                if ax is not None:
                    raise ValueError("Values were given for both `fig` and `ax`. "
                                     "Only one can be used; prefer ax.")
                warnings.warn("The `fig` keyword arugment of LivePlot is "
                              "deprecated and will be removed in the future. "
                              "Instead, use the new keyword argument `ax` to "
                              "provide specific Axes to plot on.")
                ax = fig.gca()
            if ax is None:
                fig, ax = plt.subplots()
            self.ax = ax

            if legend_keys is None:
                legend_keys = []
            self.legend_keys = ['scan_id'] + legend_keys
            if x is not None:
                self.x, *others = get_obj_fields([x])
            else:
                self.x = 'seq_num'
            self.y, *others = get_obj_fields([y])
            self.ax.set_ylabel(y)
            if (self._xlabel is None):
                self.ax.set_xlabel(x or 'sequence #')
            else:
                self.ax.set_xlabel(self._xlabel)
            if xlim is not None:
                self.ax.set_xlim(*xlim)
            if ylim is not None:
                self.ax.set_ylim(*ylim)
            self.ax.margins(.1)
            self.kwargs = kwargs
            self.lines = []
            self.legend = None
            self.legend_title = " :: ".join([name for name in self.legend_keys])
            self._epoch_offset = None  # used if x == 'time'
            self._epoch = epoch

        self.__setup = setup

    def start(self, doc):
        self.__setup()
        # The doc is not used; we just use the signal that a new run began.
        self._epoch_offset = doc['time']  # used if self.x == 'time'
        self.x_data, self.y_data = [], []
        label = " :: ".join(
            [str(doc.get(name, name)) for name in self.legend_keys])
        kwargs = ChainMap(self.kwargs, {'label': label})
        self.current_line, = self.ax.plot([], [], **kwargs)
        self.lines.append(self.current_line)
        legend = self.ax.legend(loc=0, title=self.legend_title)
        try:
            # matplotlib v3.x
            self.legend = legend.set_draggable(True)
        except AttributeError:
            # matplotlib v2.x (warns in 3.x)
            self.legend = legend.draggable(True)
        super().start(doc)

    def event(self, doc):
        "Unpack data from the event and call self.update()."
        # This outer try/except block is needed because multiple event
        # streams will be emitted by the RunEngine and not all event
        # streams will have the keys we want.
        if flyer_id_mono.flying_dev.control.control.get() != 5:
            return
        flag_use_data = False
        try:
            # This inner try/except block handles seq_num and time, which could
            # be keys in the data or accessing the standard entries in every
            # event.
            if 'descriptor' not in doc:
                return
            try:
                new_x = doc['data'][self.x]
            except KeyError:
                if self.x in ('time', 'seq_num'):
                    new_x = doc[self.x]
                else:
                    raise
            if self.y in doc['data']:
                flag_use_data = True
                new_y = doc['data'][self.y]
                # print(f'1\t{new_x=}\t{new_y=}')
            if self._y_norm in doc['data']:
                self._y_norm_val = doc['data'][self._y_norm]
                # print(f'{self._y_norm_val=}')
        except KeyError:
            # wrong event stream, skip it
            return

        if flag_use_data is False:
            return
        # Special-case 'time' to plot against against experiment epoch, not
        # UNIX epoch.
        if self.x == 'time' and self._epoch == 'run':
            new_x -= self._epoch_offset

        #overright the x value
        new_x = self._epts[self._xind]
        self._xind = self._xind + 1

        # overwrite the y value
        # if new_y_norm == 0:
        #     new_x = 0
        new_y = new_y / self._y_norm_val
        
        self.update_caches(new_x, new_y)
        self.update_plot()
        super().event(doc)

    def update_caches(self, x, y):
        if (x > 0):
            self.x_data.append(x)
            self.y_data.append(y)

    def update_plot(self):
        self.current_line.set_data(self.x_data, self.y_data)
        # Rescale and redraw.
        self.ax.relim(visible_only=True)
        self.ax.autoscale_view(tight=True)
        self.ax.figure.canvas.draw_idle()

    def stop(self, doc):
        if not self.x_data:
            print('LivePlot did not get any data that corresponds to the '
                  'x axis. {}'.format(self.x))
        if not self.y_data:
            print('LivePlot did not get any data that corresponds to the '
                  'y axis. {}'.format(self.y))
        if len(self.y_data) != len(self.x_data):
            print('LivePlot has a different number of elements for x ({}) and'
                  'y ({})'.format(len(self.x_data), len(self.y_data)))
        super().stop(doc)

# LivePlot for XANES measurements
#   Need to remove i0_baseline (not using current anymore and we won't have
#     1e-10 counts
#   Can we incorporate all the detector channels, instead of only one?
class NormalizeLivePlot(HackLivePlot):
    def __init__(self, *args, norm_key=None,  **kwargs):
        super().__init__(*args, **kwargs)
        if norm_key is None:
            raise RuntimeError("norm key is required kwarg")
        self._norm = norm_key

    def event(self, doc):
        "Update line with data from this Event."
        try:
            if self.x is not None:
                # this try/except block is needed because multiple event
                # streams will be emitted by the RunEngine and not all event
                # streams will have the keys we want
                new_x = doc['data'][self.x]
            else:
                new_x = doc['seq_num']
            new_y = doc['data'][self.y]
            new_norm = doc['data'][self._norm]
        except KeyError:
            # wrong event stream, skip it
            return
        self.y_data.append(new_y / abs(new_norm-i0_baseline))
        self.x_data.append(new_x)
        self.current_line.set_data(self.x_data, self.y_data)
        # Rescale and redraw.
        self.ax.relim(visible_only=True)
        self.ax.autoscale_view(tight=True)
        self.ax.figure.canvas.draw_idle()


# I don't think anything below here is used...
# Can be removed once this is confirmed.
def make_live_image(image_axes, key):
    """
    Example
    p--------

    fig, ax = plt.subplots()
    image_axes = ax.imshow(np.zeros((476, 512)), vmin=0, vmax=2)
    cb = make_live_image(image_axes, 'pixi_image_array_data')
    RE(Count([pixi]), subs={'event': [cb]})
    """
    def live_image(name, doc):
        if name != 'event':
            return
        image_axes.set_data(doc['data'][key].reshape(476, 512))
    return live_image


class SRXLiveImage(CallbackBase):
    """
    Stream 2D images in a cross-section viewer.

    Parameters
    ----------
    field : string name of data field in an Event

    Note
    ----
    Requires a matplotlib fix that is not released as of this writing. The
    relevant commit is a951b7.
    """
    def __init__(self, field):
        super().__init__()
        self.field = field
        fig = plt.figure()
        self.cs = CrossSection(fig)
        self.cs._fig.show()

    def event(self, doc):
        uid = doc['data'][self.field]
        data = db.reg.retrieve(uid)
        self.cs.update_image(data)
        self.cs._fig.canvas.draw_idle()
