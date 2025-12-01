import io
import numpy as np
from typing import Tuple, List

import cv2
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_tensor, to_pil_image

import tkinter as tk
from tkinter import filedialog, ttk, messagebox

from inpainting_models.pconv.unet import PConvUNet
from inpainting_models.deepfill.generators import Generator as DeepFillGenerator
from segmentation_models.maskrcnn import MaskRCNNPredictor
from segmentation_models.deeplabv3 import DeepLabV3Predictor
from utils import load_ckpt_pconv
from sentence_transformers_predictor import SentenceTransformersPredictor

# =========================================================
# Global config
# =========================================================

IMG_SIZE = (256, 256)  # PConv train size (W,H)
DEEPFILL_SIZE = (512, 512)  # DeepFill size (W,H)

# Canvas / display size
DISPLAY_MAX_W = 900
DISPLAY_MAX_H = 700


# =========================================================
# Generic helpers
# =========================================================

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_maskrcnn(device: str):
    model = MaskRCNNPredictor(device=device, score_thresh=0.5)
    return model


def load_deeplab(device: str):
    model = DeepLabV3Predictor(device=device)
    return model, model.transforms, model.categories


def load_pconv(device: str):
    model = PConvUNet(in_ch=3, layer_size=6)
    model.to(device)
    model_pre = load_ckpt_pconv("pretrained_pconv.pth", model, for_predict=True, device=device)
    model_pre.eval()
    return model_pre


def load_deepfill(device: str):
    model = DeepFillGenerator(return_flow=False, checkpoint="file.pth")
    model.to(device).eval()
    return model


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    return TF.to_tensor(pil_img.convert("RGB"))


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.clamp(0, 1)
    return TF.to_pil_image(t)


def make_display_pil(pil_img: Image.Image,
                     max_w: int = DISPLAY_MAX_W,
                     max_h: int = DISPLAY_MAX_H):
    """
    Resize PIL image for display, return (resized_pil, scale_factor).
    Allows upscaling small images to better fill the canvas.
    """
    w, h = pil_img.size
    scale = min(max_w / w, max_h / h)
    if scale != 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.BICUBIC)
    return pil_img, scale


def pil_to_tk(pil_img: Image.Image, max_w=500, max_h=500):
    """Resize PIL for thumbnail & convert to ImageTk.PhotoImage."""
    w, h = pil_img.size
    scale = min(max_w / w, max_h / h)
    if scale != 1.0:
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return ImageTk.PhotoImage(pil_img)


# =========================================================
# PConv / DeepFill helpers (your logic)
# =========================================================

def predict_pconv(model, image, mask, device):
    model.eval()
    image = image.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    with torch.no_grad():
        pred_img, pred_mask = model(image, mask)
    pred_img = pred_img.squeeze(0).cpu()
    pred_mask = pred_mask.squeeze(0).cpu()
    return pred_img, pred_mask


def inpaint_with_pconv_from_gui(pconv_model, pil_img, hole_mask_np, device):
    pil_resized = pil_img.resize(IMG_SIZE, Image.BICUBIC)
    img_t = to_tensor(pil_resized)  # [3,256,256]

    mask_small = cv2.resize(
        hole_mask_np.astype(np.uint8),
        IMG_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    valid_mask_np = (1 - mask_small).astype(np.uint8) * 255
    mask_pil = Image.fromarray(valid_mask_np, mode="L").convert("RGB")
    mask_t = to_tensor(mask_pil)

    corrupted = img_t * mask_t

    pred_img, pred_mask = predict_pconv(pconv_model, corrupted, mask_t, device)

    pred_pil = to_pil_image(pred_img.clamp(0, 1))
    pred_pil_up = pred_pil.resize(pil_img.size, Image.BICUBIC)

    corrupted_pil = to_pil_image(corrupted.clamp(0, 1))
    corrupted_pil_up = corrupted_pil.resize(pil_img.size, Image.BICUBIC)

    return pred_pil_up, corrupted_pil_up, mask_pil


def inpaint_with_deepfill_from_gui(deepfill_model, pil_img, hole_mask_np, device, size=DEEPFILL_SIZE):
    pil_resized = pil_img.resize(size, Image.BICUBIC)
    img_t = TF.to_tensor(pil_resized)
    _, H, W = img_t.shape

    mask_small = cv2.resize(
        hole_mask_np.astype(np.float32),
        size,
        interpolation=cv2.INTER_NEAREST
    )
    mask_t = torch.from_numpy(mask_small).float().unsqueeze(0)

    with torch.no_grad():
        outs = deepfill_model.infer(
            image=img_t.to(device),
            mask=mask_t.to(device),
            return_vals=["inpainted"],
            device=device,
        )

    inpainted_np = outs[0].astype("float32") / 255.0
    inpainted_t = torch.from_numpy(inpainted_np).permute(2, 0, 1)

    masked_t = img_t * (1.0 - mask_t)

    inpainted_pil = to_pil_image(inpainted_t.clamp(0, 1))
    inpainted_pil_up = inpainted_pil.resize(pil_img.size, Image.BICUBIC)

    masked_pil = to_pil_image(masked_t.clamp(0, 1))
    masked_pil_up = masked_pil.resize(pil_img.size, Image.BICUBIC)

    return inpainted_pil_up, masked_pil_up


# =========================================================
# Instance & semantic segmentation (your logic)
# =========================================================

def get_instance_mask_from_box(
        image_pil: Image.Image,
        user_box: Tuple[int, int, int, int],
        maskrcnn: MaskRCNNPredictor,
        device: str,
) -> np.ndarray:
    img_tensor = TF.to_tensor(image_pil).to(device)

    preds = maskrcnn.predict(img_tensor)
    boxes = preds["boxes"]
    scores = preds["scores"]
    masks = preds["masks"]

    H, W = image_pil.size[1], image_pil.size[0]

    if masks is None or len(boxes) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    ux1, uy1, ux2, uy2 = user_box

    def iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = max(ay2, by2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ax2 - ax1)
        area_b = (bx2 - bx1) * (bx2 - bx1)
        return inter / (area_a + area_b - inter + 1e-8)

    ious = []
    for b in boxes:
        bx1, by1, bx2, by2 = b.tolist()
        ious.append(iou((ux1, uy1, ux2, uy2), (bx1, by1, bx2, by2)))

    if len(ious) == 0 or max(ious) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    best_idx = int(np.argmax(ious))
    best_mask = masks[best_idx, 0]

    obj_mask = (best_mask.cpu().numpy() > 0.5).astype(np.uint8)
    return obj_mask


def prep_deeplab_input(image_pil: Image.Image, deeplab_tf, device: str) -> torch.Tensor:
    tf_out = deeplab_tf(image_pil).to(device)

    if tf_out.dim() == 3:
        img = tf_out.unsqueeze(0)
    elif tf_out.dim() == 4 and tf_out.shape[1] in (1, 3):
        img = tf_out
    elif tf_out.dim() >= 4:
        c, h, w = tf_out.shape[-3:]
        img = tf_out.view(-1, c, h, w)
    else:
        raise ValueError(f"Unexpected deeplab_tf output shape: {tf_out.shape}")

    if img.shape[1] != 3:
        raise ValueError(f"Expected 3 channels for DeepLab, got {img.shape}")
    return img


def get_semantic_mask_from_class(
        image_pil: Image.Image,
        deeplab,
        deeplab_tf,
        categories: List[str],
        selected_label: str,
        device: str,
) -> np.ndarray:
    img_t = deeplab_tf(image_pil).to(device)
    pred_mask = deeplab.predict(img_t)  # [h',w']

    h, w = pred_mask.shape
    H, W = image_pil.size[1], image_pil.size[0]

    if (h, w) != (H, W):
        pred_up = F.interpolate(
            pred_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(H, W),
            mode="nearest"
        ).squeeze().long()
    else:
        pred_up = pred_mask

    if selected_label not in categories:
        return np.zeros((H, W), dtype=np.uint8)

    class_idx = categories.index(selected_label)
    obj_mask = (pred_up.cpu().numpy() == class_idx).astype(np.uint8)
    return obj_mask


def get_present_classes(
        image_pil: Image.Image,
        deeplab,
        deeplab_tf,
        categories: List[str],
        device: str,
) -> List[str]:
    img_t = deeplab_tf(image_pil).to(device)
    pred_mask = deeplab.predict(img_t)
    unique_classes = torch.unique(pred_mask).cpu().tolist()
    names = [categories[c] for c in unique_classes if 0 <= c < len(categories)]
    return sorted(set(names))


# =========================================================
# Tkinter App
# =========================================================

class InpaintingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Inpainting GUI (PConv + DeepFill)")
        self.device = get_device()
        print(f"[INFO] Using device: {self.device}")

        # Models (lazy-load some of them)
        print("[INFO] Loading PConv + DeepFill...")
        self.pconv = load_pconv(self.device)
        self.deepfill = load_deepfill(self.device)
        self.maskrcnn = None
        self.deeplab = None
        self.deeplab_tf = None
        self.categories = None
        self.nlp_predictor = None
        self.nlp_predicted_class = None

        # Image / mask state
        self.orig_pil = None  # full-res PIL
        self.display_pil = None  # resized for canvas
        self.display_scale = 1.0
        self.canvas_img_tk = None

        self.mode = tk.StringVar(value="brush")  # "instance", "semantic", "brush"
        self.view_mode = tk.StringVar(value="inpainted")  # "inpainted" or "masked"

        # For drawing
        self.bbox_start = None
        self.bbox_end = None
        self.bbox_rect_id = None

        self.mask_display = None  # HxW uint8, 1 = hole, 0 = keep (display size)
        self.last_draw_point = None
        self.brush_radius = 15

        # Semantic classes
        self.present_classes = []
        self.selected_class = tk.StringVar(value="")

        # Result images (PIL)
        self.pconv_inpaint_pil = None
        self.pconv_masked_pil = None
        self.deepfill_inpaint_pil = None
        self.deepfill_masked_pil = None
        self.mask_pil = None

        # Tk image refs
        self.tk_pconv = None
        self.tk_deepfill = None
        self.tk_mask = None

        self._build_ui()

    # ---------- UI Layout ----------

    def _build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Load image button
        ttk.Button(top, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)

        # Mode selection
        ttk.Label(top, text="Mode:").pack(side=tk.LEFT, padx=(15, 2))
        for text, val in [("Instance", "instance"),
                          ("Semantic", "semantic"),
                          ("Brush", "brush"),
                          ("NLP Query", "nlp")]:
            rb = ttk.Radiobutton(
                top, text=text, value=val,
                variable=self.mode, command=self.on_mode_change
            )
            rb.pack(side=tk.LEFT)

        # Semantic controls
        self.btn_analyze = ttk.Button(top, text="Analyze Classes", command=self.analyze_classes)
        self.btn_analyze.pack(side=tk.LEFT, padx=(20, 5))

        self.class_combo = ttk.Combobox(top, textvariable=self.selected_class, state="readonly", width=20)
        self.class_combo.pack(side=tk.LEFT, padx=5)

        # NLP query input
        ttk.Label(top, text="Or type:").pack(side=tk.LEFT, padx=(20, 2))
        self.nlp_input = tk.Entry(top, width=30)
        self.nlp_input.pack(side=tk.LEFT, padx=5)
        self.nlp_input.bind("<Return>", lambda e: self.predict_from_nlp())
        ttk.Button(top, text="Predict", command=self.predict_from_nlp).pack(side=tk.LEFT, padx=2)
        self.nlp_result_label = ttk.Label(top, text="", foreground="blue")
        self.nlp_result_label.pack(side=tk.LEFT, padx=5)

        # Run button
        ttk.Button(top, text="Run Inpainting", command=self.run_inpainting).pack(side=tk.RIGHT, padx=5)

        # Main area (left = image+mask, right = results)
        main = ttk.Frame(self.root)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ---- LEFT: canvas (top) + mask thumbnail (bottom) ----
        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas for drawing
        self.canvas = tk.Canvas(left, bg="gray", width=DISPLAY_MAX_W, height=DISPLAY_MAX_H - 200)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Mask thumbnail under canvas
        mask_frame = ttk.LabelFrame(left, text="Mask (white = remove)")
        mask_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        self.lbl_mask = ttk.Label(mask_frame)
        self.lbl_mask.pack(padx=5, pady=5)

        # Bind mouse events for canvas
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # ---- RIGHT: view toggle + PConv + DeepFill ----
        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # View mode toggle
        view_frame = ttk.LabelFrame(right, text="View")
        view_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Radiobutton(
            view_frame, text="Inpainted",
            variable=self.view_mode, value="inpainted",
            command=self.update_result_views
        ).pack(anchor="w")
        ttk.Radiobutton(
            view_frame, text="Masked input",
            variable=self.view_mode, value="masked",
            command=self.update_result_views
        ).pack(anchor="w")

        # PConv result
        ttk.Label(right, text="PConv").pack()
        self.lbl_pconv = ttk.Label(right)
        self.lbl_pconv.pack(pady=(0, 10))

        # DeepFill result
        ttk.Label(right, text="DeepFill").pack()
        self.lbl_deepfill = ttk.Label(right)
        self.lbl_deepfill.pack(pady=(0, 10))

    # ---------- Image loading & display ----------

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
        )
        if not path:
            return

        pil_img = Image.open(path).convert("RGB")
        self.orig_pil = pil_img

        disp_pil, scale = make_display_pil(pil_img)
        self.display_pil = disp_pil
        self.display_scale = scale

        self.mask_display = np.zeros((disp_pil.size[1], disp_pil.size[0]), dtype=np.uint8)  # [H,W]

        self.bbox_start = None
        self.bbox_end = None
        self.bbox_rect_id = None

        self._redraw_canvas()

        # Clear previous results
        self.lbl_pconv.config(image="", text="")
        self.lbl_deepfill.config(image="", text="")
        self.lbl_mask.config(image="", text="")
        self.tk_pconv = None
        self.tk_deepfill = None
        self.tk_mask = None

        self.pconv_inpaint_pil = None
        self.pconv_masked_pil = None
        self.deepfill_inpaint_pil = None
        self.deepfill_masked_pil = None
        self.mask_pil = None

    def _redraw_canvas(self):
        if self.display_pil is None:
            return

        base = np.array(self.display_pil)  # RGB, HxWx3

        # Overlay mask in red if any
        if self.mask_display is not None and self.mask_display.sum() > 0:
            overlay = base.copy()
            mask_bool = self.mask_display.astype(bool)
            overlay[mask_bool] = [255, 0, 0]
        else:
            overlay = base

        img = Image.fromarray(overlay)
        self.canvas_img_tk = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_img_tk)

        # Re-draw bbox (if any)
        if self.bbox_start and self.bbox_end and self.mode.get() == "instance":
            x1, y1 = self.bbox_start
            x2, y2 = self.bbox_end
            self.bbox_rect_id = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline="lime", width=2
            )

    # ---------- Mode / semantic ----------

    def on_mode_change(self):
        if self.mask_display is None:
            return
        if self.mode.get() == "brush":
            self.bbox_start = None
            self.bbox_end = None
            self.bbox_rect_id = None
        elif self.mode.get() in ("instance", "semantic"):
            self.mask_display = np.zeros_like(self.mask_display)
        elif self.mode.get() == "nlp":
            self.mask_display = np.zeros_like(self.mask_display)
            self.nlp_input.focus()
        self._redraw_canvas()

    def analyze_classes(self):
        if self.orig_pil is None:
            messagebox.showwarning("Warning", "Load an image first.")
            return

        if self.deeplab is None:
            print("[INFO] Loading DeepLab...")
            self.deeplab, self.deeplab_tf, self.categories = load_deeplab(self.device)

        print("[INFO] Analyzing classes in the image...")
        classes = get_present_classes(self.orig_pil, self.deeplab, self.deeplab_tf, self.categories, self.device)
        if not classes:
            messagebox.showinfo("Info", "No semantic classes detected.")
            return

        self.present_classes = classes
        self.class_combo["values"] = classes
        self.class_combo.current(0)
        self.selected_class.set(classes[0])
        print("[INFO] Detected classes:", classes)

    # ---------- Mouse drawing ----------

    def on_mouse_down(self, event):
        if self.display_pil is None:
            return

        mode = self.mode.get()
        if mode == "instance":
            self.bbox_start = (event.x, event.y)
            self.bbox_end = (event.x, event.y)
        elif mode == "brush":
            self.last_draw_point = (event.x, event.y)
            self._paint_at(event.x, event.y)

    def on_mouse_move(self, event):
        if self.display_pil is None:
            return

        mode = self.mode.get()
        if mode == "instance":
            if self.bbox_start:
                self.bbox_end = (event.x, event.y)
                self._redraw_canvas()
        elif mode == "brush":
            if self.last_draw_point is not None:
                self._line_on_mask(self.last_draw_point, (event.x, event.y))
                self.last_draw_point = (event.x, event.y)
                self._redraw_canvas()

    def on_mouse_up(self, event):
        mode = self.mode.get()
        if mode == "instance":
            if self.bbox_start:
                self.bbox_end = (event.x, event.y)
                self._redraw_canvas()
        elif mode == "brush":
            if self.last_draw_point is not None:
                self._paint_at(event.x, event.y)
                self.last_draw_point = None
                self._redraw_canvas()

    def _paint_at(self, x, y):
        if self.mask_display is None:
            return
        cv2.circle(self.mask_display, (x, y), self.brush_radius, 1, thickness=-1)

    def _line_on_mask(self, p1, p2):
        if self.mask_display is None:
            return
        cv2.line(self.mask_display, p1, p2, 1, thickness=self.brush_radius * 2)

    def load_nlp_predictor(self):
        if self.nlp_predictor is not None:
            return
        try:
            print("[INFO] Loading NLP predictor...")
            self.nlp_predictor = SentenceTransformersPredictor(
                head_path="sentence_transformers_head.pth",
                classes_path="sentence_transformers_classes.pkl",
                device=self.device,
                model_name="all-MiniLM-L6-v2"
            )
            print("[INFO] NLP predictor loaded successfully.")
        except FileNotFoundError as e:
            messagebox.showerror("Error", f"NLP model files not found:\n{e}\n\nPlease run training first.")
            self.nlp_predictor = None

    def predict_from_nlp(self):
        query = self.nlp_input.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Enter a query first.")
            return

        if self.nlp_predictor is None:
            self.load_nlp_predictor()
            if self.nlp_predictor is None:
                return

        predicted_class, confidence = self.nlp_predictor.predict(query)
        if predicted_class is None:
            messagebox.showwarning("Warning", "Could not understand the query.")
            return

        self.nlp_predicted_class = predicted_class
        confidence_pct = confidence * 100
        result_text = f"Predicted: {predicted_class.upper()} ({confidence_pct:.1f}%)"
        self.nlp_result_label.config(text=result_text)

        if confidence < 0.7:
            response = messagebox.askyesno(
                "Low Confidence",
                f"Confidence is {confidence_pct:.1f}%.\n\nDo you want to remove '{predicted_class}'?"
            )
            if not response:
                self.nlp_predicted_class = None
                return

        print(f"[INFO] NLP prediction: {predicted_class} ({confidence_pct:.1f}%)")

    # ---------- Inpainting ----------

    def run_inpainting(self):
        if self.orig_pil is None:
            messagebox.showwarning("Warning", "Load an image first.")
            return

        mode = self.mode.get()

        # Ensure models
        if mode == "instance" and self.maskrcnn is None:
            print("[INFO] Loading Mask R-CNN...")
            self.maskrcnn = load_maskrcnn(self.device)

        if mode == "semantic":
            if self.deeplab is None or self.deeplab_tf is None:
                print("[INFO] Loading DeepLab...")
                self.deeplab, self.deeplab_tf, self.categories = load_deeplab(self.device)
            if not self.selected_class.get():
                messagebox.showwarning("Warning", "Select a semantic class first.")
                return

        if mode == "nlp":
            if self.deeplab is None or self.deeplab_tf is None:
                print("[INFO] Loading DeepLab...")
                self.deeplab, self.deeplab_tf, self.categories = load_deeplab(self.device)

        W, H = self.orig_pil.size

        # Build hole_mask_np in ORIGINAL resolution
        if mode == "instance":
            if not (self.bbox_start and self.bbox_end):
                messagebox.showwarning("Warning", "Draw a bounding box first.")
                return
            inv = 1.0 / self.display_scale
            x1, y1 = self.bbox_start
            x2, y2 = self.bbox_end
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            ux1 = int(x1 * inv)
            uy1 = int(y1 * inv)
            ux2 = int(x2 * inv)
            uy2 = int(y2 * inv)

            user_box = (ux1, uy1, ux2, uy2)
            hole_mask_np = get_instance_mask_from_box(
                image_pil=self.orig_pil,
                user_box=user_box,
                maskrcnn=self.maskrcnn,
                device=self.device,
            )

        elif mode == "brush":
            if self.mask_display is None or self.mask_display.sum() == 0:
                messagebox.showwarning("Warning", "Paint a mask first.")
                return
            hole_mask_np = cv2.resize(
                self.mask_display.astype(np.uint8),
                (W, H),
                interpolation=cv2.INTER_NEAREST
            )

        elif mode == "semantic":
            label = self.selected_class.get()
            hole_mask_np = get_semantic_mask_from_class(
                image_pil=self.orig_pil,
                deeplab=self.deeplab,
                deeplab_tf=self.deeplab_tf,
                categories=self.categories,
                selected_label=label,
                device=self.device,
            )
            if hole_mask_np.sum() == 0:
                messagebox.showinfo("Info", f"No pixels found for class '{label}'.")
                return

        elif mode == "nlp":
            if self.nlp_predicted_class is None:
                messagebox.showwarning("Warning", "Predict a class first using NLP.")
                return
            hole_mask_np = get_semantic_mask_from_class(
                image_pil=self.orig_pil,
                deeplab=self.deeplab,
                deeplab_tf=self.deeplab_tf,
                categories=self.categories,
                selected_label=self.nlp_predicted_class,
                device=self.device,
            )
            if hole_mask_np.sum() == 0:
                messagebox.showinfo("Info", f"No pixels found for class '{self.nlp_predicted_class}'.")
                return

        else:
            messagebox.showerror("Error", "Unknown mode.")
            return

        if hole_mask_np.sum() == 0:
            messagebox.showwarning("Warning", "Mask is empty (no region selected).")
            return

        # Run PConv
        print("[INFO] Running PConv...")
        pconv_pil, masked_pconv_pil, _ = inpaint_with_pconv_from_gui(
            pconv_model=self.pconv,
            pil_img=self.orig_pil,
            hole_mask_np=hole_mask_np,
            device=self.device,
        )

        # Run DeepFill
        print("[INFO] Running DeepFill...")
        deepfill_pil, masked_deepfill_pil = inpaint_with_deepfill_from_gui(
            deepfill_model=self.deepfill,
            pil_img=self.orig_pil,
            hole_mask_np=hole_mask_np,
            device=self.device,
            size=DEEPFILL_SIZE,
        )

        # Store PIL results
        self.pconv_inpaint_pil = pconv_pil
        self.pconv_masked_pil = masked_pconv_pil
        self.deepfill_inpaint_pil = deepfill_pil
        self.deepfill_masked_pil = masked_deepfill_pil

        # Build mask visualization (white where hole=1)
        mask_vis = (hole_mask_np * 255).astype(np.uint8)  # [H,W]
        mask_vis = np.stack([mask_vis] * 3, axis=-1)  # [H,W,3]
        self.mask_pil = Image.fromarray(mask_vis)

        # Update view
        self.update_result_views()
        print("[INFO] Inpainting done.")

    def update_result_views(self):
        if self.pconv_inpaint_pil is None:
            return

        if self.view_mode.get() == "inpainted":
            pconv_src = self.pconv_inpaint_pil
            deepfill_src = self.deepfill_inpaint_pil
        else:  # "masked"
            pconv_src = self.pconv_masked_pil
            deepfill_src = self.deepfill_masked_pil

        # PConv
        self.tk_pconv = pil_to_tk(pconv_src, max_w=500, max_h=400)
        self.lbl_pconv.config(image=self.tk_pconv)

        # DeepFill
        self.tk_deepfill = pil_to_tk(deepfill_src, max_w=500, max_h=400)
        self.lbl_deepfill.config(image=self.tk_deepfill)

        # Mask thumbnail under canvas (smaller)
        if self.mask_pil is not None:
            self.tk_mask = pil_to_tk(self.mask_pil, max_w=250, max_h=200)
            self.lbl_mask.config(image=self.tk_mask)


# =========================================================
# Run app
# =========================================================

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Inpainting GUI (PConv + DeepFill)")
    root.geometry("1400x800")  # wider & taller window
    root.minsize(1200, 700)  # prevent tiny squashing
    app = InpaintingApp(root)
    root.mainloop()
