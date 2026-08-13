import cv2
import numpy as np
import os
import json
from PIL import Image, ImageDraw, ImageFont
class DateDigitModifier:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.local_fonts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    def log(self, msg):
        if self.verbose:
            print(f"[DateDigitModifier] {msg}")
    def count_holes(self, thresh_crop):
        contours, hierarchy = cv2.findContours(thresh_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return 0
        holes = 0
        for h in hierarchy[0]:
            if h[3] != -1:
                holes += 1
        return holes
    def detect_weight(self, crops, platform='android'):
        if not isinstance(crops, list):
            crops = [crops]
        total_fg = sum(np.count_nonzero(c) for c in crops)
        total_area = sum(c.size for c in crops)
        ratio = total_fg / float(total_area) if total_area > 0 else 0
        if str(platform).lower() in ['ios', 'inter']:
            return "bold" if ratio >= 0.58 else "regular"
        elif str(platform).lower() in ['notosans', 'noto']:
            return "bold" if ratio >= 0.58 else "regular"
        else:
            return "bold" if ratio >= 0.58 else "semibold"
    def auto_detect_font(self, b0, b1, b2, thresh):
        candidates = [
            ('opensans', 'OpenSans-SemiBold-Static.ttf'),
            ('inter', 'Inter-Regular.ttf'),
            ('roboto', 'Roboto-Regular.ttf')
        ]
        c0 = thresh[b0[1]:b0[1]+b0[3], b0[0]:b0[0]+b0[2]]
        c1 = thresh[b1[1]:b1[1]+b1[3], b1[0]:b1[0]+b1[2]]
        c2 = thresh[b2[1]:b2[1]+b2[3], b2[0]:b2[0]+b2[2]]
        scores = {}
        for name, font_file in candidates:
            fpath = os.path.join(self.local_fonts_dir, font_file)
            if not os.path.exists(fpath):
                fpath = f"/code/fonts/{font_file}"
                if not os.path.exists(fpath):
                    continue
            def score_digit(char_crop, char_str, w, h):
                scale = 4
                cw, ch = max(4, w * scale), max(4, h * scale)
                fsize = int(ch * 1.02)
                try:
                    font = ImageFont.truetype(fpath, fsize)
                    bbox = font.getbbox(char_str)
                    th = bbox[3] - bbox[1]
                    if th > 0:
                        fsize = int(fsize * (ch / float(th)))
                        font = ImageFont.truetype(fpath, fsize)
                        bbox = font.getbbox(char_str)
                        th = bbox[3] - bbox[1]
                    tw = bbox[2] - bbox[0]
                    canvas = Image.new('L', (cw, ch), 0)
                    draw = ImageDraw.Draw(canvas)
                    draw.text(((cw - tw)//2 - bbox[0], (ch - th)//2 - bbox[1]), char_str, fill=255, font=font)
                    rend = np.array(canvas.resize((w, h), Image.Resampling.LANCZOS))
                    rend_t = rend > 127
                    crop_t = char_crop > 0
                    inter = np.count_nonzero(crop_t & rend_t)
                    union = np.count_nonzero(crop_t | rend_t)
                    return inter / float(union) if union > 0 else 0
                except Exception:
                    return 0.0
            s0 = score_digit(c0, '2', b0[2], b0[3])
            s1 = score_digit(c1, '0', b1[2], b1[3])
            s2 = score_digit(c2, '0', b2[2], b2[3])
            total_score = (s0 * 0.4) + (s1 * 0.3) + (s2 * 0.3)
            scores[name] = total_score
        if scores:
            best_name = max(scores.items(), key=lambda x: x[1])[0]
            return best_name
        return "opensans"
    def extract_colors_robust(self, img, b0, b1, b2, b3, thresh):
        x_min = max(0, b0[0] - 5)
        x_max = min(img.shape[1], b3[0] + b3[2] + 5)
        y_min = max(0, min(b0[1], b1[1], b2[1], b3[1]) - 5)
        y_max = min(img.shape[0], max(b0[1]+b0[3], b1[1]+b1[3], b2[1]+b2[3], b3[1]+b3[3]) + 5)
        patch_img = img[y_min:y_max, x_min:x_max]
        patch_thresh = thresh[y_min:y_max, x_min:x_max]
        text_mask = patch_thresh > 0
        bg_mask = patch_thresh == 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bg_safe_mask = cv2.erode(bg_mask.astype(np.uint8), kernel, iterations=1) > 0
        core_text_mask = cv2.erode(text_mask.astype(np.uint8), kernel, iterations=1) > 0
        if np.any(core_text_mask):
            text_color = np.median(patch_img[core_text_mask], axis=0)
        elif np.any(text_mask):
            text_color = np.median(patch_img[text_mask], axis=0)
        else:
            text_color = np.array([20., 20., 20.])
        if np.any(bg_safe_mask):
            bg_color = np.median(patch_img[bg_safe_mask], axis=0)
        elif np.any(bg_mask):
            bg_color = np.median(patch_img[bg_mask], axis=0)
        else:
            bg_color = np.array([255., 255., 255.])
        bg_gray = cv2.cvtColor(patch_img, cv2.COLOR_BGR2GRAY)
        bg_pixels_gray = bg_gray[bg_safe_mask] if np.any(bg_safe_mask) else bg_gray
        bg_sigma = float(np.std(bg_pixels_gray)) if bg_pixels_gray.size > 0 else 0.0
        return text_color, bg_color, bg_sigma
    def find_year_pattern(self, img, platform='android'):
        img_h, img_w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)
        thresh = cv2.bitwise_or(otsu_thresh, adaptive_thresh)
        min_h = max(6, int(img_h * 0.008))
        max_h = min(int(img_h * 0.25), max(80, int(img_h * 0.15)))
        min_w = max(3, int(img_w * 0.004))
        max_w = min(int(img_w * 0.15), max(60, int(img_w * 0.08)))
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        raw_boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if min_h <= h <= max_h and min_w <= w <= max_w:
                raw_boxes.append((x, y, w, h))
        boxes = []
        for b in raw_boxes:
            x, y, w, h = b
            is_inner = False
            for p in raw_boxes:
                px, py, pw, ph = p
                if (px < x and py < y and px + pw > x + w and py + ph > y + h):
                    is_inner = True
                    break
            if not is_inner:
                boxes.append(b)
        line_tolerance = max(4, int(img_h * 0.015))
        boxes = sorted(boxes, key=lambda b: (b[1] // line_tolerance, b[0]))
        lines = {}
        for b in boxes:
            y_group = b[1] // line_tolerance
            lines.setdefault(y_group, []).append(b)
        candidates = []
        for y_group, line_boxes in lines.items():
            line_boxes = sorted(line_boxes, key=lambda b: b[0])
            unique_line = []
            for b in line_boxes:
                if not unique_line or abs(b[0] - unique_line[-1][0]) > max(3, int(b[2] * 0.3)):
                    unique_line.append(b)
            if len(unique_line) < 4:
                continue
            for i in range(len(unique_line) - 3):
                b0, b1, b2, b3 = unique_line[i:i+4]
                dx1 = b1[0] - b0[0]
                dx2 = b2[0] - b1[0]
                dx3 = b3[0] - b2[0]
                y_diff_max = max(abs(b0[1] - b1[1]), abs(b1[1] - b2[1]), abs(b2[1] - b3[1]))
                h_diff_max = max(abs(b0[3] - b1[3]), abs(b1[3] - b2[3]), abs(b2[3] - b3[3]))
                max_allowed_y_diff = max(4, int(b0[3] * 0.25))
                max_allowed_h_diff = max(4, int(b0[3] * 0.25))
                ar0 = b0[2] / float(b0[3])
                ar1 = b1[2] / float(b1[3])
                ar2 = b2[2] / float(b2[3])
                ar3 = b3[2] / float(b3[3])
                if not (0.25 <= ar0 <= 0.95 and 0.25 <= ar1 <= 0.95 and 0.25 <= ar2 <= 0.95 and 0.25 <= ar3 <= 0.95):
                    continue
                if y_diff_max <= max_allowed_y_diff and h_diff_max <= max_allowed_h_diff:
                    avg_h = (b0[3] + b1[3] + b2[3] + b3[3]) / 4.0
                    avg_w = (b0[2] + b1[2] + b2[2] + b3[2]) / 4.0
                    if not (dx1 <= avg_h * 1.8 and dx2 <= avg_h * 1.8 and dx3 <= avg_h * 1.8):
                        continue
                    if not (dx1 >= avg_w * 0.5 and dx2 >= avg_w * 0.5 and dx3 >= avg_w * 0.5):
                        continue
                    if abs(dx1 - dx2) <= max(5, int(dx1 * 0.35)) and abs(dx2 - dx3) <= max(5, int(dx1 * 0.35)):
                        pitch = (dx1 + dx2 + dx3) / 3.0
                        has_hyphen_or_slash = False
                        x_sep_start = b3[0] + b3[2]
                        x_sep_end = min(img_w, x_sep_start + int(pitch * 1.6))
                        y_sep_start = max(0, b0[1] - 2)
                        y_sep_end = min(thresh.shape[0], b0[1] + b0[3] + 2)
                        sep_crop = thresh[y_sep_start:y_sep_end, x_sep_start:x_sep_end]
                        if sep_crop.size > 0:
                            cnts_sep, _ = cv2.findContours(sep_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for c_sep in cnts_sep:
                                xs, ys, ws, hs = cv2.boundingRect(c_sep)
                                if ws >= 2 and hs >= 1 and hs <= 0.65 * avg_h:
                                    has_hyphen_or_slash = True
                                    break
                        c0 = thresh[b0[1]:b0[1]+b0[3], b0[0]:b0[0]+b0[2]]
                        c1 = thresh[b1[1]:b1[1]+b1[3], b1[0]:b1[0]+b1[2]]
                        c2 = thresh[b2[1]:b2[1]+b2[3], b2[0]:b2[0]+b2[2]]
                        h0, h1, h2 = self.count_holes(c0), self.count_holes(c1), self.count_holes(c2)
                        detected_weight = self.detect_weight([c0, c1, c2], platform=platform)
                        score = 100
                        if has_hyphen_or_slash:
                            score += 200
                        if (h0 == 0 and h1 >= 1 and h2 >= 1):
                            score += 50
                        elif (h0 == 0 and h1 >= 1):
                            score += 30
                        text_color, bg_color, bg_sigma = self.extract_colors_robust(img, b0, b1, b2, b3, thresh)
                        candidates.append({
                            'score': score,
                            'b0': b0, 'b1': b1, 'b2': b2, 'b3': b3,
                            'pitch': pitch,
                            'avg_h': avg_h,
                            'avg_w': avg_w,
                            'detected_weight': detected_weight,
                            'text_color': text_color,
                            'bg_color': bg_color,
                            'bg_sigma': bg_sigma,
                            'all_boxes': boxes
                        })
        if candidates:
            best = sorted(candidates, key=lambda c: (c['score'], c['avg_h']), reverse=True)[0]
            b0, b1, b2, b3 = best['b0'], best['b1'], best['b2'], best['b3']
            detected_font = self.auto_detect_font(b0, b1, b2, thresh)
            best['detected_font'] = detected_font
            self.log(f"Detected 4-digit year pattern at y={b0[1]}:")
            self.log(f"  b0: {b0}")
            self.log(f"  b1: {b1}")
            self.log(f"  b2: {b2}")
            self.log(f"  b3: {b3}")
            self.log(f"  Auto-detected Font Family: {detected_font}")
            return best
        self.log("Applying adaptive document text line detection fallback...")
        if len(boxes) >= 4:
            boxes_sorted = sorted(boxes, key=lambda b: b[1])
            best_line = boxes_sorted[:4]
            b0, b1, b2, b3 = best_line[0], best_line[1], best_line[2], best_line[3]
            pitch = abs(b1[0] - b0[0]) if abs(b1[0] - b0[0]) > 5 else 20.0
            text_color, bg_color, bg_sigma = self.extract_colors_robust(img, b0, b1, b2, b3, thresh)
            return {
                'score': 10,
                'b0': b0, 'b1': b1, 'b2': b2, 'b3': b3,
                'pitch': pitch,
                'avg_h': float(b3[3]),
                'avg_w': float(b3[2]),
                'detected_weight': 'semibold',
                'text_color': text_color,
                'bg_color': bg_color,
                'bg_sigma': bg_sigma,
                'all_boxes': boxes
            }
        raise ValueError("Could not locate text/date digits in image. Please ensure the document is clear and readable.")
    def find_200_prefix(self, img):
        return self.find_year_pattern(img)
    def estimate_document_text_blur(self, img, b0, b1=None, b2=None):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boxes = [b for b in [b0, b1, b2] if b is not None]
        grads = []
        for b in boxes:
            crop = gray[b[1]:b[1]+b[3], b[0]:b[0]+b[2]]
            if crop.size == 0:
                continue
            gx = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(gx**2 + gy**2)
            edge_pixels = mag[mag > 40]
            if edge_pixels.size > 0:
                grads.append(np.mean(edge_pixels))
        mean_grad = float(np.mean(grads)) if grads else 600.0
        if mean_grad >= 580.0:
            sigma = 0.0
        else:
            sigma = min(1.2, max(0.15, (580.0 - mean_grad) / 220.0))
        return mean_grad, sigma
    def render_synthetic_digit(self, target_digit, h, w, text_color, bg_color, is_bold=False, platform='android', sigma_doc=0.0):
        p = str(platform).lower()
        if p in ['notosans', 'noto']:
            font_name = "NotoSans-Bold.ttf" if is_bold else "NotoSans-Medium.ttf"
        elif p in ['ios', 'inter']:
            font_name = "Inter-Bold.ttf" if is_bold else "Inter-Regular.ttf"
        elif p in ['roboto']:
            font_name = "Roboto-Bold.ttf" if is_bold else "Roboto-Regular.ttf"
        else:
            font_name = "OpenSans-Bold-Static.ttf" if is_bold else "OpenSans-SemiBold-Static.ttf"
        selected_font = os.path.join(self.local_fonts_dir, font_name)
        if not os.path.exists(selected_font):
            if p in ['notosans', 'noto']:
                fallback_name = "NotoSans-Medium.ttf"
            elif p in ['ios', 'inter']:
                fallback_name = "Inter-Regular.ttf"
            elif p in ['roboto']:
                fallback_name = "Roboto-Regular.ttf"
            else:
                fallback_name = "OpenSans-SemiBold-Static.ttf"
            fallback_paths = [
                os.path.join(self.local_fonts_dir, fallback_name),
                f"/code/fonts/{fallback_name}"
            ]
            for fb in fallback_paths:
                if os.path.exists(fb):
                    selected_font = fb
                    break
        scale = 4
        canvas_w = int(w * scale)
        canvas_h = int(h * scale)
        font_size = int(canvas_h * 1.02)
        try:
            font = ImageFont.truetype(selected_font, font_size) if os.path.exists(selected_font) else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        dummy = Image.new("RGBA", (canvas_w * 2, canvas_h * 2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy)
        ref_digit = "0"
        ref_bbox = draw.textbbox((0, 0), ref_digit, font=font)
        th_ref = ref_bbox[3] - ref_bbox[1]
        if th_ref > 0 and os.path.exists(selected_font):
            font_size = int(font_size * (canvas_h / float(th_ref)))
            font = ImageFont.truetype(selected_font, font_size)
            ref_bbox = draw.textbbox((0, 0), ref_digit, font=font)
            th_ref = ref_bbox[3] - ref_bbox[1]
        bbox_target = draw.textbbox((0, 0), str(target_digit), font=font)
        tw_target = bbox_target[2] - bbox_target[0]
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        tc = (int(text_color[2]), int(text_color[1]), int(text_color[0]), 255)
        tx = (canvas_w - tw_target) // 2 - bbox_target[0]
        ty = (canvas_h - th_ref) // 2 - ref_bbox[1]
        draw.text((tx, ty), str(target_digit), fill=tc, font=font)
        if sigma_doc > 0.02:
            sigma_canvas = sigma_doc * scale
            arr_canvas = np.array(canvas)
            arr_blurred = cv2.GaussianBlur(arr_canvas, (0, 0), sigmaX=sigma_canvas, sigmaY=sigma_canvas)
            canvas = Image.fromarray(arr_blurred)
        rendered = canvas.resize((w, h), Image.Resampling.LANCZOS)
        return np.array(rendered)
    def process_image(self, img_input, new_digit, manual_box=None, platform='auto'):
        if isinstance(img_input, str):
            img = cv2.imread(img_input)
            if img is None:
                raise ValueError(f"Could not read image from path: {img_input}")
        else:
            img = img_input.copy()
        new_digit_str = str(new_digit)
        if manual_box is not None:
            x3, y3, w3, h3 = manual_box
            info = {
                'b0': (x3 - int(w3*3), y3, w3, h3),
                'b1': (x3 - int(w3*2), y3, w3, h3),
                'b2': (x3 - int(w3*1), y3, w3, h3),
                'b3': (x3, y3, w3, h3),
                'pitch': float(w3),
                'avg_h': float(h3),
                'avg_w': float(w3),
                'detected_weight': 'semibold',
                'detected_font': 'opensans',
                'bg_sigma': 0.0,
                'text_color': np.array([20., 20., 20.]),
                'bg_color': np.array([255., 255., 255.]),
                'all_boxes': [(x3, y3, w3, h3)]
            }
        else:
            info = self.find_year_pattern(img, platform=platform)
        b0, b1, b2, b3 = info['b0'], info['b1'], info['b2'], info['b3']
        y_top_neighbors = [b0[1], b1[1], b2[1]]
        y_bottom_neighbors = [b0[1] + b0[3], b1[1] + b1[3], b2[1] + b2[3]]
        y3_unified = int(round(np.median(y_top_neighbors)))
        y3_bottom_unified = int(round(np.median(y_bottom_neighbors)))
        h3_unified = max(1, y3_bottom_unified - y3_unified)
        w3_unified = int(round(np.median([b0[2], b1[2], b2[2]])))
        pitch = ( (b1[0] - b0[0]) + (b2[0] - b1[0]) ) / 2.0
        x3_ideal = int(round(b0[0] + pitch * 3.0))
        x3 = x3_ideal
        y3 = y3_unified
        w3 = w3_unified
        h3 = h3_unified
        out_img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        x_erase_min = min(img.shape[1] - 1, max(b2[0] + b2[2] + 1, x3 - 2))
        x_erase_max = min(img.shape[1], x3 + w3 + 2)
        y_erase_min = max(0, y3 - 2)
        y_erase_max = min(img.shape[0], y3 + h3 + 2)
        bg_color = info['bg_color']
        erase_mask = thresh[y_erase_min:y_erase_max, x_erase_min:x_erase_max] > 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        erase_mask_dilated = cv2.dilate(erase_mask.astype(np.uint8), kernel, iterations=1) > 0
        target_patch = out_img[y_erase_min:y_erase_max, x_erase_min:x_erase_max]
        target_patch[erase_mask_dilated] = np.clip(bg_color, 0, 255).astype(np.uint8)
        out_img[y_erase_min:y_erase_max, x_erase_min:x_erase_max] = target_patch
        is_bold = (info.get('detected_weight') == 'bold')
        eff_platform = info.get('detected_font', 'opensans') if (platform in ['auto', None, '']) else platform
        self.log(f"Rendering 100% pure synthetic TrueType font (font: {eff_platform}, weight: {info.get('detected_weight')}) for digit '{new_digit_str}'")
        mean_g, sigma_b = self.estimate_document_text_blur(img, b0, b1, b2)
        self.log(f"Document edge gradient: {mean_g:.1f} -> Subpixel PSF blur sigma: {sigma_b:.4f}")
        synth_rgba = self.render_synthetic_digit(new_digit_str, h3, w3, info['text_color'], info['bg_color'], is_bold=is_bold, platform=eff_platform, sigma_doc=sigma_b)
        alpha = (synth_rgba[:, :, 3] / 255.0)[:, :, np.newaxis]
        fg_rgb = synth_rgba[:, :, :3][:, :, ::-1]
        bg_region = out_img[y3:y3+h3, x3:x3+w3].astype(float)
        blended = fg_rgb * alpha + bg_region * (1 - alpha)
        bg_sigma = info.get('bg_sigma', 0.0)
        if bg_sigma > 1.2:
            noise = np.random.normal(0, bg_sigma * 0.35, (h3, w3, 3))
            blended = blended + noise
        out_img[y3:y3+h3, x3:x3+w3] = np.clip(blended, 0, 255).astype(np.uint8)
        diff_img = out_img.copy()
        cv2.rectangle(diff_img, (x3-2, y3-2), (x3+w3+2, y3+h3+2), (0, 255, 0), 2)
        return {
            'image': out_img,
            'diff_image': diff_img,
            'bbox': (x3, y3, w3, h3),
            'method': 'pure_synthetic_opensans'
        }