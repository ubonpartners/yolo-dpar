import cairo
import numpy as np
import time
import other.coord as coord

class ARGBdraw:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        self.ctx = cairo.Context(self.surface)
        self.clear()

    def clear(self):
        """Clear the surface to fully transparent."""
        self.ctx.set_operator(cairo.OPERATOR_CLEAR)
        self.ctx.paint()
        self.ctx.set_operator(cairo.OPERATOR_OVER)

    def _set_color(self, clr):
        colours={"red":[0,0,255],
             "orange":[0,128,255],
             "green":[0,255,0],
             "cyan":[255,255,0],
             "blue":[255,0,0],
             "yellow":[0,255,255],
             "white":[255,255,255],
             "black":[0,0,0]}

        alphas={"solid":255,
            "half":128,
            "flashing":128,
            "transparent":64}

        if isinstance(clr, str):
            alpha=None
            if "_" in clr:
                alpha,clr=clr.split("_")
            assert clr in colours, f"unknown colour {clr}"
            clr=colours[clr]
            if alpha in alphas:
                clr=[alphas[alpha]]+clr
                # special case "flashing" make alpha
                # change between 0-255 based on time
                if alpha=="flashing":
                    t=int(time.time()*512) & 511
                    if t>255:
                        t=511-t
                    clr[0]=t

        if len(clr)==3:
            b,g,r=clr
            a=255
        else:
            a, b, g, r=clr

        self.ctx.set_source_rgba(r / 255.0, g / 255.0, b / 255.0, a / 255.0)

    def box(self, box, clr=(255,255,255,255), line_width=1.0):
        x=self.width*box[0]
        y=self.height*box[1]
        w=self.width*(box[2]-box[0])
        h=self.height*(box[3]-box[1])
        self._set_color(clr)
        if  line_width>0:
            self.ctx.set_line_width(line_width)
            self.ctx.rectangle(x, y, w, h)
            self.ctx.stroke()
        else:
            self.ctx.rectangle(x, y, w, h)
            self.ctx.fill()

    def line(self, start, stop, clr=(255,255,255,255), line_width=1.0):
        self._set_color(clr)
        self.ctx.set_line_width(line_width)
        x1=self.width*start[0]
        y1=self.height*start[1]
        x2=self.width*stop[0]
        y2=self.height*stop[1]

        self.ctx.move_to(x1, y1)
        self.ctx.line_to(x2, y2)
        self.ctx.stroke()

    def circle(self, c, radius, clr=(255,255,255,255), fill=True, line_width=1.0):
        cx=self.width*c[0]
        cy=self.height*c[1]
        radius=radius*self.height
        self._set_color(clr)
        self.ctx.arc(cx, cy, radius, 0, 2 * np.pi)
        if fill:
            self.ctx.fill()
        else:
            self.ctx.set_line_width(line_width)
            self.ctx.stroke()

    def text(self, text, pos, clr=(255, 255, 255, 255), bg_clr=None, font_size=16, font_face="monospace"):
        x = self.width * pos[0]
        y = self.height * pos[1]

        self.ctx.select_font_face(font_face, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        self.ctx.set_font_size(font_size)

        lines = text.split('\n')
        line_extents = [self.ctx.text_extents(line) for line in lines]

        # Compute bounding box for all lines if background color is set
        if bg_clr is not None:
            padding = 2
            line_spacing = font_size * 1.2  # approximate line height
            max_width = max(ext.width for ext in line_extents)
            total_height = line_spacing * len(lines)

            # Use the first line extents for bearing reference
            first_ext = line_extents[0]
            box_x = x + first_ext.x_bearing - padding
            box_y = y + first_ext.y_bearing - padding
            box_w = max_width + 2 * padding
            box_h = total_height + 2 * padding

            self._set_color(bg_clr)
            self.ctx.rectangle(box_x, box_y, box_w, box_h)
            self.ctx.fill()

        self._set_color(clr)
        line_spacing = font_size * 1.2
        for i, line in enumerate(lines):
            self.ctx.move_to(x, y + i * line_spacing)
            self.ctx.show_text(line)

    def get_scaled_numpy_view(self, width, height):
        """
        Returns a scaled (height, width, 4) uint8 ARGB NumPy array of the surface.
        The result is still premultiplied ARGB.
        """
        # Create a new surface of target size
        scaled_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(scaled_surface)

        # Set up scaling transform
        scale_x = width / self.width
        scale_y = height / self.height
        ctx.scale(scale_x, scale_y)

        # Paint the original surface onto the scaled one
        ctx.set_source_surface(self.surface, 0, 0)
        ctx.paint()

        # Extract data from scaled surface
        buf = scaled_surface.get_data()
        bgra = np.ndarray((self.height, self.width, 4), dtype=np.uint8, buffer=buf)
        argb = bgra[..., [3, 2, 1, 0]]
        return argb

    def get_numpy_view(self):
        """Returns a (H, W, 4) uint8 NumPy array in ARGB format (premultiplied)."""
        buf = self.surface.get_data()
        bgra = np.ndarray((self.height, self.width, 4), dtype=np.uint8, buffer=buf)
        argb = bgra[..., [3, 2, 1, 0]]
        return argb

def blend_argb_over_rgb(argb, rgb):
    """
    Blend a premultiplied ARGB image over an RGB background.

    Parameters:
    - argb: (H, W, 4) ARGB premultiplied uint8 array
    - rgb: (H, W, 3) uint8 background

    Returns:
    - result: (H, W, 3) uint8 composited RGB image
    """
    alpha = argb[..., 0:1].astype(np.float32) / 255.0
    rgb_fg = argb[..., 1:].astype(np.float32)
    rgb_bg = rgb.astype(np.float32)

    out_rgb = rgb_fg + rgb_bg * (1 - alpha)
    return np.clip(out_rgb, 0, 255).astype(np.uint8)
