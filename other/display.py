import cv2
import numpy as np
import other.coord as coord
import other.ARGBdraw as ARGBdraw
import other.ultralytics as sultralytics

def window_mouse_callback(event, x, y, flags, display):
    xc=(x-display.pad_l)/display.img_width
    yc=(y-display.pad_t)/display.img_height
    #print(xc,yc)
    if xc<0 or xc>1 or yc<0 or yc>1:
        return
    if event==cv2.EVENT_LBUTTONDOWN or event==cv2.EVENT_MOUSEMOVE:
        boxes=display.selected_boxes([xc,yc])
        lbutton=event==cv2.EVENT_LBUTTONDOWN
        event={"x":xc,
               "y":yc,
               "key":None,
               "lbutton":lbutton,
               "rbutton":False,
               "selected":boxes}
        display.events.append(event)

class Display:
    def __init__(self, width=1920, height=1080, image=None, name="noname", output=None):
        self.width=width
        self.height=height
        self.window_name=name
        self.events=[]
        self.selected_boxes_list=[]
        self.overlay=ARGBdraw.ARGBdraw(self.width, self.height)
        if image is None:
            image=np.zeros((self.height, self.width, 3), np.uint8)

        self.writer=None
        if output is not None:
            print(f"Writing display output to {output} with {width}x{height}x30fps")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(output, fourcc, 30.0, (width, height))

        self.show(image)

        cv2.imshow(self.window_name, image)
        cv2.setMouseCallback(self.window_name, window_mouse_callback, self)

    def close(self):
        print("Destroying display")
        if self.writer is not None:
            self.writer.release()
            self.writer=None
        if self.window_name is not None:
            cv2.destroyWindow(self.window_name)
            self.window_name=None

    def __del__(self):
        self.close()

    def selected_boxes(self, pt):
        best_boxes=[]
        for b in self.selected_boxes_list:
            d=coord.point_in_box(pt, b["box"])
            if d is not None:
                bc=b.copy()
                bc["dist"]=d
                best_boxes.append(bc)
        best_boxes.sort(key=lambda x: x["dist"])
        return best_boxes

    def show(self, image, title=None, is_rgb=False):
        h, w, _ = image.shape
        scale=min(self.width/w, self.height/h)
        self.img_width=min(self.width, int(scale*w))
        self.img_height=min(self.height, int(scale*h))
        self.pad_l=(self.width-self.img_width)//2
        self.pad_t=(self.height-self.img_height)//2
        self.pad_r=self.width-self.img_width-self.pad_l
        self.pad_b=self.height-self.img_height-self.pad_t

        self.img_roi=[self.pad_l/self.width,
                      self.pad_t/self.height,
                      1.0-self.pad_r/self.width,
                      1.0-self.pad_b/self.height]

        if is_rgb:
            image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_resized=cv2.resize(image, (self.img_width, self.img_height))
        image_padded=cv2.copyMakeBorder(image_resized,
                                        self.pad_t, self.pad_b, self.pad_l, self.pad_r,
                                        cv2.BORDER_CONSTANT, (0,0,0))

        argb=self.overlay.get_numpy_view()
        result=ARGBdraw.blend_argb_over_rgb(argb, image_padded)

        cv2.imshow(self.window_name, result)
        if title is not None:
            cv2.setWindowTitle(self.window_name, title)

        if self.writer is not None:
            self.writer.write(result)

    def get_events(self, delay_ms):
        r=cv2.waitKey(delay_ms)  # Press any key to move to the next image
        if r==27:
            print("Quitting")
            quit()
        if r!=-1:
            event={"x":None, "y":None, "key":chr(r), "lbutton":False, "rbutton":False}
            self.events.append(event)
        ret=self.events
        self.events=[]
        return ret

    def clear(self):
        self.overlay.clear()
        self.selected_boxes_list=[]

    def draw_line(self, start, stop, clr=None, thickness=1):
        start_img=coord.unmap_roi_point(self.img_roi, start)
        stop_img=coord.unmap_roi_point(self.img_roi, stop)
        self.overlay.line(start_img, stop_img, clr=clr, line_width=thickness)

    def draw_box(self, box, clr=None, thickness=1, select_context=None):
        box_img=coord.unmap_roi_box(self.img_roi, box)
        self.overlay.box(box_img, clr=clr, line_width=thickness)
        if select_context:
            self.selected_boxes_list.append({"box":box, "context":select_context})

    def draw_circle(self, centre, radius, clr=None, thickness=1):
        c=coord.unmap_roi_point(self.img_roi, centre)
        self.overlay.circle(c, radius, clr=clr, line_width=thickness)

    def draw_text(self, text, xc, yc,
              font=cv2.FONT_HERSHEY_SIMPLEX,
              fontScale=0.75,
              fontColor=(255,255,255,255),
              bgColor=(128,0,0,0),
              lineType=2,
              thickness=1,
              unmap=True):
        if unmap:
            pos_img=coord.unmap_roi_point(self.img_roi, [xc,yc])
        else:
            pos_img=[xc,yc]

        self.overlay.text(text, pos_img, clr=fontColor, bg_clr=bgColor)

def display_image_wait_key(image, scale=0, title="no title"):
    display=Display(image=image, name=title)
    events=display.get_events(0)
    key=None
    for e in events:
        if e["key"]!=None:
            key=e
    del display
    return key

def faf_display(list_of_stuff, title="no title"):
    img=None
    objs=[]
    for i, s in enumerate(list_of_stuff):
        if isinstance(s,np.ndarray):
            img=s
        if isinstance(s, dict):
            objs.append(s)
        if isinstance(s, list):
            for s2 in s:
                if isinstance(s2, dict):
                    objs.append(s2)

    display=Display(width=1280, height=720)
    display.clear
    display.show(img, title=title)
    sultralytics.draw_boxes(display, objs)
    display.show(img, title=title)
    while True:
        events=display.get_events(10)
        key=None
        for e in events:
            if e["key"]!=None:
                key=e
        if key is not None:
            break
    del display
