from __future__ import annotations
import cv2
import numpy as np

_MAX_DISPLAY_W = 900   # scale down large sensors for the selector window
_FRAME_MS      = 33    # ~30 fps live update interval


class ROISelector:
    """
    Live-feed ROI selector via Tkinter.

    The camera keeps streaming so you can physically position it while
    drawing the rectangle.  Tkinter talks to X11 directly and bypasses
    the Qt layer that crashes on Pi OS Bookworm.

    Controls:
        Drag left mouse   — draw selection rectangle
        ENTER / SPACE     — confirm
        R                 — clear and redraw
        Q / Escape        — cancel

    Requires:
        sudo apt install -y python3-tk python3-pil.imagetk
    """

    def select(self, cam) -> tuple[int, int, int, int] | None:
        try:
            import tkinter as tk
        except ImportError:
            self._hint("python3-tk not found", "sudo apt install -y python3-tk")
            return None
        try:
            from PIL import Image, ImageTk
        except ImportError:
            self._hint("Pillow ImageTk not found",
                       "sudo apt install -y python3-pil.imagetk")
            return None

        # ---- measure frame size from first grab ---------------------
        for _ in range(3):
            cam.read()
        ret, frame0 = cam.read()
        if not ret:
            print("[ERROR] Cannot read camera frame.")
            return None

        orig_h, orig_w = frame0.shape[:2]
        scale  = min(1.0, _MAX_DISPLAY_W / orig_w)
        disp_w = int(orig_w * scale)
        disp_h = int(orig_h * scale)

        # ---- shared mutable state -----------------------------------
        roi_out  = [None]
        start_xy = [None]
        rect_id  = [None]
        live     = [True]   # False once the user confirms/cancels

        # ---- build window -------------------------------------------
        root = tk.Tk()
        root.title("Live ROI selection  |  Drag → ENTER confirm  |  R reset  |  Q cancel")
        root.resizable(False, False)

        canvas = tk.Canvas(root, width=disp_w, height=disp_h,
                           cursor="crosshair", highlightthickness=0, bg="black")
        canvas.pack()

        status = tk.Label(root,
                          text="Position camera, then drag to select the print region",
                          fg="#00cc44", bg="#1e1e1e", font=("Helvetica", 11), pady=4)
        status.pack(fill=tk.X)

        img_item  = canvas.create_image(0, 0, anchor=tk.NW)
        photo_ref = [None]   # must hold a Python reference or GC collects it

        # ---- live frame update loop ---------------------------------
        def update_frame():
            if not live[0]:
                return
            ret, frame = cam.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if scale < 1.0:
                    rgb = cv2.resize(rgb, (disp_w, disp_h))
                photo_ref[0] = ImageTk.PhotoImage(Image.fromarray(rgb))
                canvas.itemconfig(img_item, image=photo_ref[0])
                # Keep the selection rectangle on top of the new frame
                if rect_id[0]:
                    canvas.tag_raise(rect_id[0])
            root.after(_FRAME_MS, update_frame)

        # ---- mouse callbacks ----------------------------------------
        def on_press(e):
            start_xy[0] = (e.x, e.y)
            if rect_id[0]:
                canvas.delete(rect_id[0])
                rect_id[0] = None

        def on_drag(e):
            if not start_xy[0]:
                return
            if rect_id[0]:
                canvas.delete(rect_id[0])
            rect_id[0] = canvas.create_rectangle(
                start_xy[0][0], start_xy[0][1], e.x, e.y,
                outline="#00ff55", width=2,
            )

        def on_release(e):
            if not start_xy[0]:
                return
            x1 = min(start_xy[0][0], e.x)
            y1 = min(start_xy[0][1], e.y)
            x2 = max(start_xy[0][0], e.x)
            y2 = max(start_xy[0][1], e.y)
            if (x2 - x1) >= 16 and (y2 - y1) >= 16:
                roi_out[0] = (
                    int(x1 / scale),
                    int(y1 / scale),
                    int((x2 - x1) / scale),
                    int((y2 - y1) / scale),
                )
                status.config(
                    text=f"ROI {roi_out[0][2]}×{roi_out[0][3]} px  — ENTER to confirm | R to redraw"
                )

        def confirm(e=None):
            if roi_out[0]:
                live[0] = False
                root.quit()

        def reset(e=None):
            roi_out[0] = None
            start_xy[0] = None
            if rect_id[0]:
                canvas.delete(rect_id[0])
                rect_id[0] = None
            status.config(text="Position camera, then drag to select the print region")

        def cancel(e=None):
            live[0] = False
            roi_out[0] = None
            root.quit()

        canvas.bind("<ButtonPress-1>",   on_press)
        canvas.bind("<B1-Motion>",        on_drag)
        canvas.bind("<ButtonRelease-1>",  on_release)
        root.bind("<Return>",             confirm)
        root.bind("<space>",              confirm)
        root.bind("r",                    reset)
        root.bind("R",                    reset)
        root.bind("q",                    cancel)
        root.bind("Q",                    cancel)
        root.bind("<Escape>",             cancel)

        update_frame()
        root.mainloop()
        try:
            root.destroy()
        except Exception:
            pass

        if roi_out[0]:
            x, y, w, h = roi_out[0]
            print(f"[INFO] ROI selected: x={x} y={y} w={w} h={h}")

        return roi_out[0]

    @staticmethod
    def _hint(problem: str, fix: str) -> None:
        print(f"[ERROR] {problem}.")
        print(f"        {fix}")
        print("        Or skip the GUI:  python main.py --roi X Y W H")
