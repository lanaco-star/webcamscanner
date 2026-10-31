"""
PaperScan - webcam document scanner for Windows.
Live preview -> auto page detection -> corner adjustment -> dewarp -> clean up -> PDF.

Install:
    pip install opencv-python pillow numpy
Run:
    python paperscan.py
Build a standalone exe:
    pyinstaller --noconsole --onefile --name PaperScan paperscan.py
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

MAX_LONG_SIDE = 2400
FILTERS = ["color", "gray", "bw", "raw"]


# ----------------------------------------------------------------- helpers
def order_quad(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def detect_quad(frame):
    """Return TL,TR,BR,BL quad of the largest page-like rectangle, or None."""
    h, w = frame.shape[:2]
    k = 800.0 / max(h, w)
    small = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    edges = cv2.erode(edges, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:8]
    min_area = 0.15 * small.shape[0] * small.shape[1]
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_quad(approx / k)
    return None


def warp(img, quad):
    tl, tr, br, bl = quad
    W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if W < 20 or H < 20:
        return img.copy()
    long_side = max(W, H)
    if long_side > MAX_LONG_SIDE:
        f = MAX_LONG_SIDE / long_side
        W, H = int(W * f), int(H * f)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC)


def remove_shadows(bgr):
    out = []
    for ch in cv2.split(bgr):
        bg = cv2.GaussianBlur(ch, (0, 0), 15)
        bg = np.where(bg == 0, 1, bg).astype(np.float32)
        out.append(np.clip(ch.astype(np.float32) / bg * 200.0, 0, 255).astype(np.uint8))
    return cv2.merge(out)


def white_balance(bgr):
    f = bgr.astype(np.float32)
    means = np.maximum(f.reshape(-1, 3).mean(axis=0), 1.0)
    f *= means.mean() / means
    return np.clip(f, 0, 255).astype(np.uint8)


def apply_filter(bgr, name):
    if name == "raw":
        return bgr
    flat = remove_shadows(bgr)
    if name == "color":
        return cv2.convertScaleAbs(white_balance(flat), alpha=1.25, beta=8)
    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    if name == "gray":
        g = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    block = max(15, (min(gray.shape) // 40) | 1)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 12
    )
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


# ----------------------------------------------------------------- app
class App:
    def __init__(self, root):
        self.root = root
        root.title("PaperScan - Webcam Document Scanner")
        root.geometry("1240x800")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = None
        self.mode = "live"          # live | adjust | page
        self.frame = None
        self.frozen = None
        self.corners = None
        self.pages = []             # [{"raw": ndarray, "filter": str}]
        self.sel = None
        self.drag = None
        self.scale = 1.0
        self.off = (0, 0)
        self.photo = None

        self.build_ui()
        root.bind("<space>", lambda e: self.capture())
        root.bind("<Return>", lambda e: self.accept())
        root.bind("<Escape>", lambda e: self.cancel())
        root.after(30, self.tick)

    # ---- ui
    def build_ui(self):
        bar = ttk.Frame(self.root, padding=8)
        bar.pack(side="left", fill="y")

        ttk.Label(bar, text="Camera").pack(anchor="w")
        row = ttk.Frame(bar)
        row.pack(anchor="w", pady=(0, 8))
        self.cam_idx = tk.IntVar(value=0)
        ttk.Spinbox(row, from_=0, to=4, width=4, textvariable=self.cam_idx).pack(side="left")
        ttk.Button(row, text="Start", command=self.start_cam).pack(side="left", padx=2)
        ttk.Button(row, text="Stop", command=self.stop_cam).pack(side="left")

        ttk.Button(bar, text="Capture  (Space)", command=self.capture).pack(fill="x", pady=2)
        ttk.Button(bar, text="Auto-detect edges", command=self.redetect).pack(fill="x", pady=2)
        ttk.Button(bar, text="Warp & add page  (Enter)", command=self.accept).pack(fill="x", pady=2)
        ttk.Button(bar, text="Cancel  (Esc)", command=self.cancel).pack(fill="x", pady=2)

        ttk.Separator(bar).pack(fill="x", pady=8)
        ttk.Label(bar, text="Enhance selected page").pack(anchor="w")
        self.filter = tk.StringVar(value="color")
        cb = ttk.Combobox(bar, textvariable=self.filter, values=FILTERS, state="readonly")
        cb.pack(fill="x", pady=(0, 8))
        cb.bind("<<ComboboxSelected>>", lambda e: self.reapply())

        ttk.Separator(bar).pack(fill="x", pady=8)
        ttk.Label(bar, text="Pages").pack(anchor="w")
        self.listbox = tk.Listbox(bar, height=10, exportselection=False)
        self.listbox.pack(fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        ttk.Button(bar, text="Delete page", command=self.delete_page).pack(fill="x", pady=2)

        ttk.Separator(bar).pack(fill="x", pady=8)
        ttk.Button(bar, text="Save PDF", command=self.save_pdf).pack(fill="x", pady=2)
        ttk.Button(bar, text="Save images", command=self.save_images).pack(fill="x", pady=2)
        ttk.Button(bar, text="Clear all", command=self.clear_all).pack(fill="x", pady=2)

        self.canvas = tk.Canvas(self.root, bg="#141414", highlightthickness=0)
        self.canvas.pack(side="right", fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "drag", None))

        self.status = ttk.Label(self.root, anchor="w", padding=4)
        self.status.pack(side="bottom", fill="x")

    # ---- camera
    def start_cam(self):
        self.stop_cam()
        cap = cv2.VideoCapture(self.cam_idx.get(), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            messagebox.showerror("Camera", "Could not open camera.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap = cap
        self.mode = "live"
        self.status.config(text="Live - place the page in view, then press Space.")

    def stop_cam(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def tick(self):
        if self.mode == "live" and self.cap is not None:
            ok, frame = self.cap.read()
            if ok:
                self.frame = frame
                quad = detect_quad(frame)
                self.show(frame, quad)
                self.status.config(
                    text="Page detected - press Space." if quad is not None
                    else "No page detected - press Space to place corners by hand."
                )
        self.root.after(30, self.tick)

    # ---- display
    def show(self, bgr, quad=None, handles=False):
        h, w = bgr.shape[:2]
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        s = min(cw / w, ch / h)
        dw, dh = max(1, int(w * s)), max(1, int(h * s))
        img = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA)
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        self.scale, self.off = s, (ox, oy)

        if quad is not None:
            pts = [(int(x * s + ox), int(y * s + oy)) for x, y in quad]
            cv2.polylines(img, [np.array(pts, np.int32)], True, (0, 220, 90), 2)
            if handles:
                for x, y in pts:
                    cv2.rectangle(img, (x - 8, y - 8), (x + 8, y + 8), (0, 220, 90), -1)

        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=self.photo)

    # ---- capture / adjust
    def capture(self):
        if self.mode == "adjust":
            return
        if self.frame is None:
            messagebox.showinfo("Camera", "Start the camera first.")
            return
        self.frozen = self.frame.copy()
        self.mode = "adjust"
        self.redetect()

    def redetect(self):
        if self.mode != "adjust":
            return
        h, w = self.frozen.shape[:2]
        quad = detect_quad(self.frozen)
        if quad is None:
            m = 0.1
            quad = np.array(
                [[w * m, h * m], [w * (1 - m), h * m], [w * (1 - m), h * (1 - m)], [w * m, h * (1 - m)]],
                np.float32,
            )
        self.corners = quad
        self.show(self.frozen, self.corners, handles=True)
        self.status.config(text="Drag the green corners onto the page, then press Enter.")

    def to_frame(self, cx, cy):
        ox, oy = self.off
        return (cx - ox) / self.scale, (cy - oy) / self.scale

    def on_press(self, e):
        if self.mode != "adjust":
            return
        fx, fy = self.to_frame(e.x, e.y)
        best, bd = None, 1e9
        for i, (x, y) in enumerate(self.corners):
            d = (x - fx) ** 2 + (y - fy) ** 2
            if d < bd:
                best, bd = i, d
        if bd < (30 / self.scale) ** 2:
            self.drag = best

    def on_drag(self, e):
        if self.mode != "adjust" or self.drag is None:
            return
        fx, fy = self.to_frame(e.x, e.y)
        h, w = self.frozen.shape[:2]
        self.corners[self.drag] = (min(max(fx, 0), w - 1), min(max(fy, 0), h - 1))
        self.show(self.frozen, self.corners, handles=True)

    def accept(self):
        if self.mode != "adjust":
            return
        page = warp(self.frozen, self.corners)
        self.pages.append({"raw": page, "filter": self.filter.get()})
        self.mode = "live"
        self.corners = None
        self.sel = len(self.pages) - 1
        self.refresh_list()
        self.show(apply_filter(page, self.filter.get()))
        self.status.config(text=f"{len(self.pages)} page(s) - press Space for the next one.")

    def cancel(self):
        if self.mode == "adjust":
            self.mode = "live"
            self.corners = None
            self.status.config(text="Capture cancelled.")

    # ---- pages
    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        for i, p in enumerate(self.pages):
            self.listbox.insert(tk.END, f"Page {i + 1}  [{p['filter']}]")
        if self.sel is not None and self.sel < len(self.pages):
            self.listbox.selection_set(self.sel)
            self.filter.set(self.pages[self.sel]["filter"])

    def on_select(self, _):
        sel = self.listbox.curselection()
        if not sel:
            return
        self.sel = sel[0]
        self.mode = "page"
        self.filter.set(self.pages[self.sel]["filter"])
        self.show(apply_filter(self.pages[self.sel]["raw"], self.pages[self.sel]["filter"]))

    def reapply(self):
        if self.sel is None or self.sel >= len(self.pages):
            return
        self.pages[self.sel]["filter"] = self.filter.get()
        self.refresh_list()
        self.show(apply_filter(self.pages[self.sel]["raw"], self.filter.get()))

    def delete_page(self):
        if self.sel is None or self.sel >= len(self.pages):
            return
        self.pages.pop(self.sel)
        self.sel = min(self.sel, len(self.pages) - 1) if self.pages else None
        self.refresh_list()
        if self.sel is not None:
            p = self.pages[self.sel]
            self.show(apply_filter(p["raw"], p["filter"]))
        else:
            self.canvas.delete("all")

    def clear_all(self):
        if self.pages and not messagebox.askyesno("Clear", "Discard all pages?"):
            return
        self.pages.clear()
        self.sel = None
        self.listbox.delete(0, tk.END)
        self.canvas.delete("all")

    # ---- export
    def save_pdf(self):
        if not self.pages:
            messagebox.showinfo("Export", "No pages yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("PDF", "*.pdf")], initialfile="scan.pdf"
        )
        if not path:
            return
        imgs = [
            Image.fromarray(cv2.cvtColor(apply_filter(p["raw"], p["filter"]), cv2.COLOR_BGR2RGB))
            for p in self.pages
        ]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], resolution=200.0)
        self.status.config(text=f"Saved {path}")

    def save_images(self):
        if not self.pages:
            messagebox.showinfo("Export", "No pages yet.")
            return
        folder = filedialog.askdirectory()
        if not folder:
            return
        for i, p in enumerate(self.pages, 1):
            out = apply_filter(p["raw"], p["filter"])
            cv2.imwrite(os.path.join(folder, f"page_{i:03d}.png"), out)
        self.status.config(text=f"Saved {len(self.pages)} image(s) to {folder}")

    def on_close(self):
        self.stop_cam()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    ttk.Style().theme_use("vista" if os.name == "nt" else "default")
    App(root)
    root.mainloop()