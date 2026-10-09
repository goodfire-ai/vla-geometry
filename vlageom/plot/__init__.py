"""Plot style and colormap helpers shared by the figure scripts (see `vlageom.plot.style`)."""
from . import style
from .style import (FIELD_CMAP, GRAPHITE, GRIPPER_CLOSE, GRIPPER_OPEN, INK, META, META_LINE,
                    PRIMARY, SLATE, VIOLET, apply_style, body_family, cmap_bridge, cmap_spec,
                    finish, register_managua_red, save, setup_floor)

__all__ = [
    "FIELD_CMAP", "GRAPHITE", "GRIPPER_CLOSE", "GRIPPER_OPEN", "INK", "META", "META_LINE",
    "PRIMARY", "SLATE", "VIOLET", "apply_style", "body_family", "cmap_bridge", "cmap_spec",
    "finish", "register_managua_red", "save", "setup_floor", "style",
]
