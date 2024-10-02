# Taken from https://github.com/pwwang/plotnine-prism/blob/master/plotnine_prism/scale.py
import json
from functools import lru_cache
from pathlib import Path
from typing import Union

import plotnine as p9
from plotnine import theme


try:
    from plotnine.scales.scale_discrete import scale_discrete
except ImportError:
    from plotnine.scales.scale import scale_discrete

# from plotnine.scales
#

SCHEMES_DIR = Path(__file__).parent / "schemes"


@lru_cache()
def list_themes():
    """List all available theme palettes"""
    return [
        tfile.stem
        for tfile in SCHEMES_DIR.glob("*.json")
        if not tfile.stem.startswith("_")
    ]


@lru_cache()
def _all_color_pals():
    with SCHEMES_DIR.joinpath("_color_palettes.json").open() as fcolor:
        return json.load(fcolor)


@lru_cache()
def _all_fill_pals():
    with SCHEMES_DIR.joinpath("_fill_palettes.json").open() as ffill:
        return json.load(ffill)


@lru_cache()
def _all_shape_pals():
    with SCHEMES_DIR.joinpath("_shape_palettes.json").open() as fshape:
        return json.load(fshape)


def list_color_pals():
    """List all available color palettes"""
    return list(_all_color_pals())


def list_fill_pals():
    """List all available fill palettes"""
    return list(_all_fill_pals())


def list_shape_pals():
    """List all available shape palettes"""
    return list(_all_shape_pals())


def prism_color_pal(palette):
    """Get the prism color palette by name"""
    return lambda n: _all_color_pals()[palette][:n]


def prism_fill_pal(palette):
    """Get the prism fill palette by name"""
    return lambda n: _all_fill_pals().get(palette, _all_color_pals()[palette])[:n]


def prism_shape_pal(palette):
    """Get the prism shape palette by name"""
    return lambda n: _all_shape_pals()[palette][:n]


class scale_color_prism(scale_discrete):
    """Prism color scale
    Args:
        palette: The color palette name
    """

    _aesthetics = ["color"]
    na_value = "#7F7F7F"

    def __init__(self, palette="colors", **kwargs):
        """Construct"""
        self.palette = prism_color_pal(palette)
        scale_discrete.__init__(self, **kwargs)


class scale_fill_prism(scale_color_prism):
    """Prism fill scale
    Args:
        palette: The fill palette name
    """

    _aesthetics = ["fill"]
    na_value = "#7F7F7F"

    def __init__(self, palette="colors", **kwargs):
        """Construct"""
        self.palette = prism_fill_pal(palette)
        scale_discrete.__init__(self, **kwargs)


class scale_shape_prism(scale_discrete):
    """Prism shape scale
    Args:
        palette: The shape palette name
    """

    _aesthetics = ["shape"]

    def __init__(self, palette="default", **kwargs):
        """Construct"""
        self.palette = prism_shape_pal(palette)
        scale_discrete.__init__(self, **kwargs)


def theme_ipsum(
    base_family="sans-serif",
    base_size=11.5,
    plot_title_family=None,
    plot_title_size=18,
    plot_title_face="bold",
    plot_title_margin=10,
    subtitle_family=None,
    subtitle_size=12,
    subtitle_face="plain",
    subtitle_margin=15,
    strip_text_family=None,
    strip_text_size=12,
    strip_text_face="plain",
    caption_family=None,
    caption_size=9,
    caption_face="italic",
    caption_margin=10,
    axis_text_size=None,
    axis_title_family=None,
    axis_title_size=9,
    axis_title_face="plain",
    axis_title_just="rt",
    plot_margin=(0.2,),
    grid_col="#cccccc",
    grid: Union[str, bool] = True,
    axis_col="#cccccc",
    axis=False,
    ticks=False,
):
    if plot_title_family is None:
        plot_title_family = base_family

    if subtitle_family is None:
        subtitle_family = base_family

    if strip_text_family is None:
        strip_text_family = base_family

    if caption_family is None:
        caption_family = base_family

    if axis_text_size is None:
        axis_text_size = axis_text_size

    if axis_title_family is None:
        axis_title_family = subtitle_family

    ret = p9.theme_minimal(base_family=base_family, base_size=base_size)
    ret += p9.theme(panel_border=p9.element_rect(fill="None", color="#B3B3B3", size=1))

    ret += p9.theme(legend_background=p9.element_blank())
    ret += p9.theme(legend_key=p9.element_blank())

    if isinstance(grid, str) or grid:
        ret += p9.theme(panel_grid=p9.element_line(color=grid_col, size=0.3))
        ret += p9.theme(panel_grid_major=p9.element_line(color=grid_col, size=0.4))
        ret += p9.theme(panel_grid_minor=p9.element_line(color=grid_col, size=0.3))

        if isinstance(grid, str):
            if "X" not in grid:
                ret += p9.theme(panel_grid_major_x=p9.element_blank())
            if "Y" not in grid:
                ret += p9.theme(panel_grid_major_y=p9.element_blank())
            if "x" not in grid:
                ret += p9.theme(panel_grid_minor_x=p9.element_blank())
            if "y" not in grid:
                ret += p9.theme(panel_grid_minor_y=p9.element_blank())
    else:
        ret += p9.theme(panel_grid=p9.element_blank())

    if not ticks:
        ret += p9.theme(axis_ticks=p9.element_blank())
        ret += p9.theme(axis_ticks_major_x=p9.element_blank())
        ret += p9.theme(axis_ticks_major_y=p9.element_blank())
    else:
        ret += p9.theme(axis_ticks=p9.element_line(size=0.15))
        ret += p9.theme(axis_ticks_major_x=p9.element_line(size=0.15))
        ret += p9.theme(axis_ticks_major_y=p9.element_line(size=0.15))
        ret += p9.theme(axis_ticks_length=5)
    #
    xj = {"b": 0, "l": 0, "m": 0.5, "c": 0.5, "r": 1, "t": 1}.get(
        axis_title_just[0].lower(), 0
    )
    yj = {"b": 0, "l": 0, "m": 0.5, "c": 0.5, "r": 1, "t": 0}.get(
        axis_title_just[1].lower(), 0
    )

    # print(xj, yj)

    ret += p9.theme(axis_text_x=p9.element_text(size=axis_text_size, margin=dict(t=0)))
    ret += p9.theme(axis_text_y=p9.element_text(size=axis_text_size, margin=dict(r=0)))
    ret += p9.theme(
        axis_title=p9.element_text(size=axis_title_size, family=axis_title_family)
    )
    # ret += p9.theme(
    #     axis_title_x=p9.element_text(
    #         hjust=10,
    #         size=axis_title_size,
    #         family=axis_title_family,
    #         face=axis_title_face,
    #     )
    # )
    #
    # ret += p9.theme(
    #     axis_title_y=p9.element_text(
    #         hjust=yj,
    #         size=axis_title_size,
    #         family=axis_title_family,
    #         face=axis_title_face,
    #     )
    # )
    # ret += p9.theme(
    #     axis_title_y=p9.element_text(
    #         hjust=yj,
    #         size=axis_title_size,
    #         angle=90,
    #         family=axis_title_family,
    #         face=axis_title_face,
    #     )
    # )
    ret += p9.theme(
        strip_text=p9.element_text(
            # hjust=1,
            # ha="right",
            size=strip_text_size,
            face=strip_text_face,
            family=strip_text_family,
        )
    )

    ret += p9.theme(panel_spacing=0.2)
    #
    ret += p9.theme(
        plot_title=p9.element_text(
            hjust=0,
            size=plot_title_size,
            margin=dict(b=plot_title_margin),
            family=plot_title_family,
            face=plot_title_face,
        )
    )
    # # Not supported
    # # ret += p9.theme(
    # #     plot_caption=p9.element_text(
    # #         hjust=0,
    # #         size=subtitle_size,
    # #         margin=dict(b=subtitle_margin),
    # #         family=subtitle_family,
    # #         face=subtitle_face,
    # #     )
    # # )
    ret += p9.theme(
        plot_caption=p9.element_text(
            hjust=1,
            size=caption_size,
            margin=dict(t=caption_margin),
            family=caption_family,
            face=caption_face,
        )
    )
    #
    # ret += p9.theme(plot_margin=plot_margin[0])
    #
    return ret


def scale_color_and_fill_formal():
    return scale_fill_prism("formal") + scale_color_prism("formal")


def theme_formal(
    base_family="Times",
    base_size=18,
    strip_text_size=18,
    axis_text_size=None,
    axis_title_size=None,
    grid_col="#B0B0B0",
    grid: Union[str, bool] = "XxYy",
    ticks=True,
):
    ret = p9.theme_linedraw(base_family=base_family, base_size=base_size)
    ret += p9.theme(
        strip_background=p9.element_blank(),
        strip_text_x=p9.element_text(color="black", size=strip_text_size),
        strip_text_y=p9.element_text(color="black", size=strip_text_size, angle=-90),
    )
    ret += p9.theme(
        panel_border=p9.element_rect(fill="None", color="#0F0F0F", size=1.5)
    )

    if isinstance(grid, str) or grid:
        ret += p9.theme(panel_grid=p9.element_line(color=grid_col, size=1))
        ret += p9.theme(panel_grid_major=p9.element_line(color=grid_col, size=1.75))
        ret += p9.theme(panel_grid_minor=p9.element_line(color=grid_col, size=1))

        if isinstance(grid, str):
            if "X" not in grid:
                ret += p9.theme(panel_grid_major_x=p9.element_blank())
            if "Y" not in grid:
                ret += p9.theme(panel_grid_major_y=p9.element_blank())
            if "x" not in grid:
                ret += p9.theme(panel_grid_minor_x=p9.element_blank())
            if "y" not in grid:
                ret += p9.theme(panel_grid_minor_y=p9.element_blank())
    else:
        ret += p9.theme(panel_grid=p9.element_blank())

    if not ticks:
        ret += p9.theme(axis_ticks=p9.element_blank())
        ret += p9.theme(axis_ticks_major_x=p9.element_blank())
        ret += p9.theme(axis_ticks_major_y=p9.element_blank())
    else:
        ret += p9.theme(axis_ticks_major_x=p9.element_line(size=0.5))
        ret += p9.theme(axis_ticks_major_y=p9.element_line(size=0.5))
        ret += p9.theme(axis_ticks_length=5)

    ret += p9.theme(axis_text_x=p9.element_text(size=axis_text_size, margin=dict(t=10)))
    ret += p9.theme(axis_text_y=p9.element_text(size=axis_text_size, margin=dict(r=10)))

    ret += p9.theme(
        axis_title_x=p9.element_text(margin=dict(t=15), size=axis_title_size),
        axis_title_y=p9.element_text(margin=dict(r=15), size=axis_title_size),
    )

    return ret


class theme_matplotlib_rc(theme):
    def __init__(self, rc=None):
        theme.__init__(self)
        if rc:
            self._rcParams.update(rc)
