"""
PaperScan - webcam document scanner for Windows
Improved capture, page detection, perspective correction, image enhancement,
multi-page management, and A4 PDF export.

Install:
    python -m pip install opencv-python pillow numpy

Run:
    python paperscan.py

Build a standalone Windows EXE:
    python -m pip install pyinstaller
    pyinstaller --noconsole --onefile --name PaperScan paperscan.py

Notes:
- Supports built-in and external USB webcams. Choose the camera index and backend
  in the app. Camera resolution depends on the hardware and driver.
- Perspective correction flattens a page photographed at an angle. It does not
  perform true curved-page dewarping.
"""

import os
import math
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


APP_TITLE = "PaperScan - Webcam Document Scanner"
MAX_OUTPUT_LONG_SIDE = 4000
PREVIEW_DETECT_LONG_SIDE = 1000
FILTERS = ["Color", "Grayscale", "Black & White", "Original"]
PDF_DPI = 300
A4_PORTRAIT = (2480, 3508)  # 300 DPI


# ---------------------------------------------------------------------------
# Geometry and document detection
# ---------------------------------------------------------------------------

def order_quad(points):
    """Return four points in TL, TR, BR, BL order."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]

    # Rotate so the top-left point is first.
    start = int(np.argmin(pts.sum(axis=1)))
    pts = np.roll(pts, -start, axis=0)

    # Ensure the next point is top-right rather than bottom-left.
    if pts[1, 0] < pts[-1, 0]:
        pts = pts[[0, 3, 2, 1]]

    return pts.astype(np.float32)


def _quad_dimensions(quad):
    tl, tr, br, bl = quad
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    return float(width), float(height)


def _is_reasonable_quad(quad, image_shape):
    h, w = image_shape[:2]
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    image_area = float(h * w)
    if image_area <= 0 or area < image_area * 0.08:
        return False

    qw, qh = _quad_dimensions(quad)
    if min(qw, qh) < 40:
        return False

    ratio = max(qw, qh) / max(1.0, min(qw, qh))
    # Exclude extremely thin rectangles and likely frame edges.
    if ratio > 3.2:
        return False

    margin = max(3, int(min(w, h) * 0.003))
    if np.any(quad[:, 0] < -margin) or np.any(quad[:, 0] > w - 1 + margin):
        return False
    if np.any(quad[:, 1] < -margin) or np.any(quad[:, 1] > h - 1 + margin):
        return False
    return True


def detect_quad(frame):
    """
    Detect a likely document quadrilateral.
    Returns TL, TR, BR, BL coordinates in the ORIGINAL frame, or None.
    Uses multiple edge/threshold passes and scores candidate contours.
    """
    if frame is None or frame.size == 0:
        return None

    original_h, original_w = frame.shape[:2]
    scale = min(1.0, PREVIEW_DETECT_LONG_SIDE / float(max(original_h, original_w)))
    if scale < 1.0:
        small = cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    else:
        small = frame.copy()

    h, w = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates = []
    passes = []

    # Edge pass.
    edges = cv2.Canny(gray, 40, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    passes.append(edges)

    # Adaptive threshold pass helps with low-contrast paper edges.
    block = max(15, (min(h, w) // 18) | 1)
    block = min(block, 61)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block, 7
    )
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
    passes.append(adaptive)
    passes.append(255 - adaptive)

    image_area = float(h * w)

    for binary in passes:
        contours, _ = cv2.findContours(
            binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:35]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * 0.08:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            for epsilon_factor in (0.018, 0.025, 0.035):
                approx = cv2.approxPolyDP(
                    contour, epsilon_factor * perimeter, True
                )
                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue

                quad = order_quad(approx.reshape(4, 2))
                if not _is_reasonable_quad(quad, small.shape):
                    continue

                quad_area = abs(cv2.contourArea(quad))
                area_score = quad_area / image_area

                # Prefer large, reasonably rectangular candidates.
                qw, qh = _quad_dimensions(quad)
                ratio = max(qw, qh) / max(1.0, min(qw, qh))
                ratio_penalty = 0.0 if 1.15 <= ratio <= 2.2 else 0.08
                score = area_score - ratio_penalty
                candidates.append((score, quad))
                break

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0][1]
    return (best / scale).astype(np.float32)


def warp_document(image, quad):
    """Perspective-correct the selected page without unnecessary upscaling."""
    quad = order_quad(quad)
    width, height = _quad_dimensions(quad)
    out_w = max(1, int(round(width)))
    out_h = max(1, int(round(height)))

    longest = max(out_w, out_h)
    if longest > MAX_OUTPUT_LONG_SIDE:
        factor = MAX_OUTPUT_LONG_SIDE / float(longest)
        out_w = max(1, int(round(out_w * factor)))
        out_h = max(1, int(round(out_h * factor)))

    destination = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    return cv2.warpPerspective(
        image, matrix, (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


# ---------------------------------------------------------------------------
# Image enhancement
# ---------------------------------------------------------------------------

def illumination_correct(bgr, strength=0.75):
    """
    Correct broad lighting variation on a cropped page.
    Keeps the correction deliberately moderate to avoid halos and overprocessing.
    """
    if bgr is None or bgr.size == 0:
        return bgr

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)

    min_dim = min(lightness.shape[:2])
    sigma = max(15.0, min_dim * 0.035)
    background = cv2.GaussianBlur(
        lightness, (0, 0), sigmaX=sigma, sigmaY=sigma
    ).astype(np.float32)

    safe_background = np.maximum(background, 8.0)
    target = float(np.median(background))
    corrected = lightness.astype(np.float32) / safe_background * target
    corrected = np.clip(corrected, 0, 255)

    # Blend with original luminance to avoid an artificial appearance.
    corrected = (
        lightness.astype(np.float32) * (1.0 - strength)
        + corrected * strength
    )
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge((corrected, a, b)), cv2.COLOR_LAB2BGR)


def enhance_color(bgr):
    # Conservative correction: preserve natural colour and avoid strong local
    # contrast, which can exaggerate paper texture and camera noise.
    flat = illumination_correct(bgr, strength=0.35)
    return flat


def enhance_gray(bgr):
    # Grayscale is the detail-preserving mode. Avoid CLAHE by default because
    # it can amplify paper texture and sensor noise in ordinary webcam images.
    flat = illumination_correct(bgr, strength=0.45)
    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    return gray


def enhance_bw(bgr, threshold_c=6, noise_level=1, contrast=1.0, sharpness=0):
    """
    Conservative document thresholding.
    threshold_c: adaptive threshold offset (higher = more white background).
    noise_level: 0 = none, 1 = light, 2 = moderate denoising.
    contrast: gentle contrast multiplier around mid-gray.
    sharpness: 0 = off, 1 = subtle unsharp mask.
    """
    flat = illumination_correct(bgr, strength=0.45)
    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)

    if noise_level == 1:
        gray = cv2.fastNlMeansDenoising(gray, None, h=3,
                                        templateWindowSize=7,
                                        searchWindowSize=21)
    elif noise_level >= 2:
        gray = cv2.fastNlMeansDenoising(gray, None, h=6,
                                        templateWindowSize=7,
                                        searchWindowSize=21)

    # Keep contrast adjustment mild and user-controlled.
    gray_f = gray.astype(np.float32)
    gray = np.clip((gray_f - 127.5) * float(contrast) + 127.5,
                   0, 255).astype(np.uint8)

    if sharpness:
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        gray = cv2.addWeighted(gray, 1.12, blur, -0.12, 0)

    min_dim = min(gray.shape[:2])
    block = max(25, min(75, int(min_dim / 24)))
    if block % 2 == 0:
        block += 1

    # No pre-threshold blur: it softens fine strokes and can merge characters.
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        int(threshold_c)
    )
    return bw


def apply_filter(bgr, mode, bw_c=6, bw_noise=1,
                 bw_contrast=1.0, bw_sharpness=0):
    if mode == "Original":
        return bgr.copy()
    if mode == "Color":
        return enhance_color(bgr)
    if mode == "Grayscale":
        gray = enhance_gray(bgr)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mode == "Black & White":
        bw = enhance_bw(
            bgr, threshold_c=bw_c, noise_level=bw_noise,
            contrast=bw_contrast, sharpness=bw_sharpness
        )
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    return bgr.copy()


# ---------------------------------------------------------------------------
# PDF page layout
# ---------------------------------------------------------------------------

def image_for_a4_pdf(bgr):
    """Place a page on a correctly sized A4 canvas without stretching it."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    source = Image.fromarray(rgb).convert("RGB")

    portrait_w, portrait_h = A4_PORTRAIT
    if source.width > source.height:
        canvas_size = (portrait_h, portrait_w)
    else:
        canvas_size = (portrait_w, portrait_h)

    canvas = Image.new("RGB", canvas_size, "white")
    margin = 90  # 9 mm at 300 DPI
    max_w = canvas_size[0] - margin * 2
    max_h = canvas_size[1] - margin * 2

    ratio = min(max_w / source.width, max_h / source.height)
    new_size = (
        max(1, int(round(source.width * ratio))),
        max(1, int(round(source.height * ratio)))
    )
    source = source.resize(new_size, Image.Resampling.LANCZOS)
    x = (canvas_size[0] - source.width) // 2
    y = (canvas_size[1] - source.height) // 2
    canvas.paste(source, (x, y))
    return canvas


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class PaperScanApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1360x850")
        self.root.minsize(1000, 650)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = None
        self.frame = None
        self.frozen = None
        self.corners = None
        self.pages = []  # each: {"raw": BGR ndarray, "filter": str}
        self.selected_page = None
        self.mode = "live"  # live, adjust, page
        self.drag_index = None
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        self.photo = None
        self.last_detected_quad = None
        self.camera_info = ""

        self.build_ui()
        self.root.bind("<space>", lambda _e: self.capture())
        self.root.bind("<Return>", lambda _e: self.accept_page())
        self.root.bind("<Escape>", lambda _e: self.cancel_capture())
        self.root.after(80, self.tick)

    def build_ui(self):
        panel = ttk.Frame(self.root, padding=10)
        panel.pack(side="left", fill="y")

        ttk.Label(panel, text="Camera", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        camera_row = ttk.Frame(panel)
        camera_row.pack(fill="x", pady=(5, 3))

        ttk.Label(camera_row, text="Index").pack(side="left")
        self.camera_index = tk.IntVar(value=0)
        ttk.Spinbox(camera_row, from_=0, to=9, width=4,
                    textvariable=self.camera_index).pack(side="left", padx=(5, 8))

        ttk.Label(panel, text="Camera backend").pack(anchor="w", pady=(5, 0))
        self.backend = tk.StringVar(value="Auto")
        ttk.Combobox(
            panel, textvariable=self.backend,
            values=["Auto", "DirectShow", "Media Foundation"],
            state="readonly"
        ).pack(fill="x", pady=(3, 6))

        cam_buttons = ttk.Frame(panel)
        cam_buttons.pack(fill="x")
        ttk.Button(cam_buttons, text="Start camera",
                   command=self.start_camera).pack(side="left", fill="x", expand=True)
        ttk.Button(cam_buttons, text="Stop",
                   command=self.stop_camera).pack(side="left", padx=(5, 0))

        ttk.Separator(panel).pack(fill="x", pady=10)
        ttk.Label(panel, text="Capture", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Button(panel, text="Capture page  (Space)",
                   command=self.capture).pack(fill="x", pady=3)
        ttk.Button(panel, text="Detect page edges again",
                   command=self.redetect).pack(fill="x", pady=3)
        ttk.Button(panel, text="Add corrected page  (Enter)",
                   command=self.accept_page).pack(fill="x", pady=3)
        ttk.Button(panel, text="Cancel capture  (Esc)",
                   command=self.cancel_capture).pack(fill="x", pady=3)

        ttk.Label(
            panel,
            text="Tip: drag the four green handles onto the paper corners.",
            wraplength=225
        ).pack(anchor="w", pady=(3, 5))

        ttk.Separator(panel).pack(fill="x", pady=10)
        ttk.Label(panel, text="Enhancement", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.filter_var = tk.StringVar(value="Color")
        self.filter_box = ttk.Combobox(
            panel, textvariable=self.filter_var, values=FILTERS, state="readonly"
        )
        self.filter_box.pack(fill="x", pady=(5, 3))
        self.filter_box.bind("<<ComboboxSelected>>", lambda _e: self.reapply_filter())

        ttk.Label(panel, text="B&W tuning").pack(anchor="w", pady=(5, 0))
        self.bw_c_var = tk.IntVar(value=6)
        self.bw_noise_var = tk.IntVar(value=1)
        self.bw_contrast_var = tk.DoubleVar(value=100.0)
        self.bw_sharp_var = tk.IntVar(value=0)

        self._add_scale(panel, "Background / threshold", self.bw_c_var, 2, 14)
        self._add_scale(panel, "Noise removal", self.bw_noise_var, 0, 2)
        self._add_scale(panel, "Contrast (%)", self.bw_contrast_var, 85, 125)
        self._add_scale(panel, "Sharpness", self.bw_sharp_var, 0, 1)

        ttk.Button(panel, text="Apply B&W settings",
                   command=self.reapply_filter).pack(fill="x", pady=(2, 5))

        ttk.Separator(panel).pack(fill="x", pady=10)
        ttk.Label(panel, text="Scanned pages", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.page_list = tk.Listbox(panel, height=9, exportselection=False)
        self.page_list.pack(fill="both", expand=True, pady=5)
        self.page_list.bind("<<ListboxSelect>>", self.select_page)
        ttk.Button(panel, text="Delete selected page",
                   command=self.delete_page).pack(fill="x", pady=2)

        ttk.Separator(panel).pack(fill="x", pady=10)
        ttk.Button(panel, text="Save PDF (A4, 300 DPI)",
                   command=self.save_pdf).pack(fill="x", pady=3)
        ttk.Button(panel, text="Save page images",
                   command=self.save_images).pack(fill="x", pady=3)
        ttk.Button(panel, text="Clear all pages",
                   command=self.clear_pages).pack(fill="x", pady=3)

        self.canvas = tk.Canvas(self.root, bg="#171717", highlightthickness=0)
        self.canvas.pack(side="right", fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda _e: setattr(self, "drag_index", None))

        self.status = ttk.Label(self.root, text="Choose a camera index and press Start camera.",
                                anchor="w", padding=6)
        self.status.pack(side="bottom", fill="x")

    def _add_scale(self, parent, label, variable, minimum, maximum):
        ttk.Label(parent, text=label).pack(anchor="w")
        scale = ttk.Scale(
            parent, from_=minimum, to=maximum, orient="horizontal",
            command=lambda value, var=variable: var.set(
                int(round(float(value))) if isinstance(var, tk.IntVar)
                else round(float(value), 2)
            )
        )
        scale.set(variable.get())
        scale.pack(fill="x", pady=(0, 3))

    def render_filter(self, bgr, mode):
        contrast = float(self.bw_contrast_var.get()) / 100.0
        return apply_filter(
            bgr, mode,
            bw_c=self.bw_c_var.get(),
            bw_noise=self.bw_noise_var.get(),
            bw_contrast=contrast,
            bw_sharpness=self.bw_sharp_var.get()
        )

    # ---- camera
    def _backend_value(self):
        choice = self.backend.get()
        if choice == "DirectShow":
            return cv2.CAP_DSHOW
        if choice == "Media Foundation":
            return cv2.CAP_MSMF
        return None

    def start_camera(self):
        self.stop_camera()
        index = int(self.camera_index.get())
        backend = self._backend_value()

        try:
            cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                messagebox.showerror(
                    "Camera",
                    f"Could not open camera index {index} using {self.backend.get()}.\n"
                    "Try another index or camera backend."
                )
                return

            # Request full-HD; actual resolution is read back because webcams vary.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 30)

            # Warm up the camera and read its actual frame dimensions.
            ok, test_frame = False, None
            for _ in range(8):
                ok, test_frame = cap.read()
                if ok and test_frame is not None:
                    break

            if not ok:
                cap.release()
                messagebox.showerror("Camera", "Camera opened but did not return an image.")
                return

            self.cap = cap
            self.frame = test_frame
            h, w = test_frame.shape[:2]
            self.camera_info = f"{w} × {h}"
            self.mode = "live"
            self.status.config(
                text=f"Camera {index} connected ({self.camera_info}). "
                     "Place a page in view and press Space."
            )
        except Exception as exc:
            messagebox.showerror("Camera error", str(exc))

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.status.config(text="Camera stopped.")

    def tick(self):
        if self.mode == "live" and self.cap is not None:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                self.frame = frame
                quad = detect_quad(frame)
                self.last_detected_quad = quad
                self.show_image(frame, quad=quad)
                if quad is not None:
                    self.status.config(
                        text=f"Page detected ({self.camera_info}). Press Space to capture."
                    )
                else:
                    self.status.config(
                        text=f"No clear page boundary detected ({self.camera_info}). "
                             "You can still capture and adjust corners manually."
                    )
        self.root.after(80, self.tick)

    # ---- display
    def show_image(self, bgr, quad=None, handles=False):
        if bgr is None or bgr.size == 0:
            return

        h, w = bgr.shape[:2]
        canvas_w = max(self.canvas.winfo_width(), 100)
        canvas_h = max(self.canvas.winfo_height(), 100)
        scale = min(canvas_w / w, canvas_h / h)
        display_w = max(1, int(round(w * scale)))
        display_h = max(1, int(round(h * scale)))

        display = cv2.resize(
            bgr, (display_w, display_h),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        )
        offset_x = (canvas_w - display_w) // 2
        offset_y = (canvas_h - display_h) // 2
        self.display_scale = scale
        self.display_offset = (offset_x, offset_y)

        if quad is not None:
            points = [
                (int(round(x * scale)), int(round(y * scale)))
                for x, y in quad
            ]
            cv2.polylines(
                display, [np.array(points, dtype=np.int32)],
                True, (40, 230, 100), 2, cv2.LINE_AA
            )
            if handles:
                for x, y in points:
                    cv2.circle(display, (x, y), 9, (20, 220, 90), -1, cv2.LINE_AA)
                    cv2.circle(display, (x, y), 11, (255, 255, 255), 2, cv2.LINE_AA)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.photo)

    # ---- capture and corner adjustment
    def capture(self):
        if self.mode == "adjust":
            return
        if self.frame is None:
            messagebox.showinfo("Camera", "Start a camera first.")
            return

        self.frozen = self.frame.copy()
        self.mode = "adjust"
        self.redetect()

    def redetect(self):
        if self.mode != "adjust" or self.frozen is None:
            return

        quad = detect_quad(self.frozen)
        if quad is None:
            h, w = self.frozen.shape[:2]
            # Manual starting guide only; user must position corners on the paper.
            margin_x, margin_y = w * 0.10, h * 0.10
            quad = np.array(
                [
                    [margin_x, margin_y],
                    [w - margin_x, margin_y],
                    [w - margin_x, h - margin_y],
                    [margin_x, h - margin_y]
                ],
                dtype=np.float32
            )
            self.status.config(
                text="Automatic detection failed. Move all four handles onto the paper corners."
            )
        else:
            self.status.config(
                text="Check the green corners. Drag any handle to the paper edge, then press Enter."
            )

        self.corners = order_quad(quad)
        self.show_image(self.frozen, self.corners, handles=True)

    def _canvas_to_frame(self, x, y):
        ox, oy = self.display_offset
        return (x - ox) / self.display_scale, (y - oy) / self.display_scale

    def on_mouse_down(self, event):
        if self.mode != "adjust" or self.corners is None:
            return
        fx, fy = self._canvas_to_frame(event.x, event.y)
        best_index, best_distance = None, float("inf")
        for i, (x, y) in enumerate(self.corners):
            distance = (x - fx) ** 2 + (y - fy) ** 2
            if distance < best_distance:
                best_index, best_distance = i, distance

        tolerance = max(18.0, 18.0 / max(self.display_scale, 0.01))
        if best_distance <= tolerance ** 2:
            self.drag_index = best_index

    def on_mouse_drag(self, event):
        if self.mode != "adjust" or self.drag_index is None:
            return
        fx, fy = self._canvas_to_frame(event.x, event.y)
        h, w = self.frozen.shape[:2]
        self.corners[self.drag_index] = (
            min(max(fx, 0), w - 1),
            min(max(fy, 0), h - 1)
        )
        self.show_image(self.frozen, self.corners, handles=True)

    def accept_page(self):
        if self.mode != "adjust" or self.frozen is None or self.corners is None:
            return

        try:
            page = warp_document(self.frozen, self.corners)
            if page.shape[0] < 40 or page.shape[1] < 40:
                messagebox.showwarning("Capture", "The selected page area is too small.")
                return

            selected_filter = self.filter_var.get()
            self.pages.append({"raw": page, "filter": selected_filter})
            self.selected_page = len(self.pages) - 1
            self.mode = "live"
            self.corners = None
            self.refresh_page_list()
            self.show_image(self.render_filter(page, selected_filter))
            self.status.config(
                text=f"Added page {len(self.pages)}. Press Space to capture another page."
            )
        except Exception as exc:
            messagebox.showerror("Page correction error", str(exc))

    def cancel_capture(self):
        if self.mode == "adjust":
            self.mode = "live"
            self.corners = None
            self.status.config(text="Capture cancelled.")

    # ---- page management
    def refresh_page_list(self):
        self.page_list.delete(0, tk.END)
        for index, page in enumerate(self.pages, start=1):
            self.page_list.insert(
                tk.END, f"Page {index}  [{page['filter']}]"
            )
        if self.selected_page is not None and 0 <= self.selected_page < len(self.pages):
            self.page_list.selection_set(self.selected_page)
            self.page_list.activate(self.selected_page)
            self.filter_var.set(self.pages[self.selected_page]["filter"])

    def select_page(self, _event=None):
        selection = self.page_list.curselection()
        if not selection:
            return
        self.selected_page = selection[0]
        self.mode = "page"
        page = self.pages[self.selected_page]
        self.filter_var.set(page["filter"])
        self.show_image(self.render_filter(page["raw"], page["filter"]))

    def reapply_filter(self):
        if self.selected_page is None or not (0 <= self.selected_page < len(self.pages)):
            return
        page = self.pages[self.selected_page]
        page["filter"] = self.filter_var.get()
        self.refresh_page_list()
        self.show_image(self.render_filter(page["raw"], page["filter"]))

    def delete_page(self):
        if self.selected_page is None or not (0 <= self.selected_page < len(self.pages)):
            return
        self.pages.pop(self.selected_page)
        if self.pages:
            self.selected_page = min(self.selected_page, len(self.pages) - 1)
            self.refresh_page_list()
            page = self.pages[self.selected_page]
            self.show_image(self.render_filter(page["raw"], page["filter"]))
        else:
            self.selected_page = None
            self.page_list.delete(0, tk.END)
            self.canvas.delete("all")
        self.mode = "page" if self.pages else "live"

    def clear_pages(self):
        if self.pages and not messagebox.askyesno(
            "Clear pages", "Delete all scanned pages from this session?"
        ):
            return
        self.pages.clear()
        self.selected_page = None
        self.page_list.delete(0, tk.END)
        self.canvas.delete("all")
        self.mode = "live"

    # ---- export
    def save_pdf(self):
        if not self.pages:
            messagebox.showinfo("Export PDF", "There are no scanned pages yet.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF document", "*.pdf")],
            initialfile="PaperScan.pdf"
        )
        if not path:
            return

        try:
            pdf_pages = []
            for page in self.pages:
                enhanced = self.render_filter(page["raw"], page["filter"])
                pdf_pages.append(image_for_a4_pdf(enhanced))

            pdf_pages[0].save(
                path,
                "PDF",
                save_all=True,
                append_images=pdf_pages[1:],
                resolution=PDF_DPI
            )
            self.status.config(text=f"Saved {len(pdf_pages)} A4 page(s) to {path}")
            messagebox.showinfo("PDF saved", f"Saved {len(pdf_pages)} page(s):\n{path}")
        except Exception as exc:
            messagebox.showerror("PDF export error", str(exc))

    def save_images(self):
        if not self.pages:
            messagebox.showinfo("Save images", "There are no scanned pages yet.")
            return

        folder = filedialog.askdirectory(title="Choose a folder for scanned images")
        if not folder:
            return

        try:
            for index, page in enumerate(self.pages, start=1):
                enhanced = self.render_filter(page["raw"], page["filter"])
                output_path = os.path.join(folder, f"PaperScan_page_{index:03d}.png")
                if not cv2.imwrite(output_path, enhanced):
                    raise OSError(f"Could not write {output_path}")

            self.status.config(text=f"Saved {len(self.pages)} image(s) to {folder}")
            messagebox.showinfo("Images saved", f"Saved {len(self.pages)} image(s) to:\n{folder}")
        except Exception as exc:
            messagebox.showerror("Image export error", str(exc))

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if os.name == "nt" and "vista" in style.theme_names():
            style.theme_use("vista")
    except tk.TclError:
        pass

    PaperScanApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
