
@parameter_annotation_decorator({
    "parameters": {
        "dets": {"default": "['xs2', 'sclr1']"},
    }
})
def xs2_1d_scan(stage, x0, x1, nx, dwell, dets=[xs, sclr1], shutter=True):
    yield from check_shutters(shutter, 'Open')
    yield from mov(xs.external_trig, False)
    yield from mov(xs.cam.acquire_time, dwell)
    yield from mov(xs.total_points, nx)
    yield from mov(xs.cam.num_images, nx)
    yield from mov(sclr1.preset_time, dwell)
    yield from subs_wrapper(
        scan(dets, stage, x0, x1, nx),
        {
            "all": [
                LivePlot(
                    "xs_channel08_mcaroi01_total_rbv",
                    x=stage.name,
                )
            ]
        },
    )
    yield from check_shutters(shutter, 'Close')


@parameter_annotation_decorator({
    "parameters": {
        "dets": {"default": "['xs2', 'sclr1']"},
    }
})
def xs2_1d_relscan(stage, neg_dx, pos_dx, nx, dwell, dets=[xs, sclr1], shutter=True):
    yield from check_shutters(shutter, 'Open')
    yield from mov(xs.external_trig, False)
    yield from mov(xs.cam.acquire_time, dwell)
    yield from mov(xs.total_points, nx)
    yield from mov(xs.cam.num_images, nx)
    yield from mov(sclr1.preset_time, dwell)
    yield from subs_wrapper(
        rel_scan(dets, stage, neg_dx, pos_dx, nx),
        {
            "all": [
                LivePlot(
                    "xs_channel08_mcaroi01_total_rbv",
                    x=stage.name,
                )
            ]
        },
    )
    yield from check_shutters(shutter, 'Close')

