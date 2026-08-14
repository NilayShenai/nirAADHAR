# nirAADHAR

Deterministic computer vision engine and web service for typography-accurate digit modification and quality matching in identity document images.

---

## Overview

`nirAADHAR` is an automated image processing pipeline designed to detect 4-digit date patterns in scanned or photographed documents, erase target numerals without spatial residue, and synthesize replacement digits that match the original typography, stroke width, baseline alignment, and camera Point Spread Function (PSF).

The engine operates fully in memory without intermediate disk writes or database persistence.

---

## Architecture & Processing Pipeline

```
  Input Image (BGR)
         │
         ▼
  Dual-Pass Otsu Thresholding & Contour Hierarchy Extraction
         │
         ▼
  4-Digit Geometric Cluster Locator & Candidate Scoring
         │
         ▼
  Multi-Glyph Shape Correlation (Auto Font Family Classification)
         │
         ▼
  Anchor Neighbor Baseline & Pitch Normalization (b0, b1, b2)
         │
         ▼
  Boundary-Clamped Morphological Stroke Erasure
         │
         ▼
  4× Supersampled TrueType Rasterization & Subpixel PSF Blur
         │
         ▼
  Lanczos-3 Downsampling & Subpixel Alpha Blending
         │
         ▼
  Output Image (PNG Bytes / Zero Disk I/O)
```

---

## Mathematical Formulation & Computer Vision Methods

### 1. Adaptive Binarization & Contour Filtering

The image $I \in \mathbb{R}^{H \times W \times 3}$ is converted to single-channel luminance $I_{\text{gray}}$ and binarized via Otsu's thresholding:

$$\tau^* = \arg\max_{\tau} \sigma_B^2(\tau) = \omega_0(\tau)\omega_1(\tau)\left[\mu_0(\tau) - \mu_1(\tau)\right]^2$$

Contour extraction with topological hierarchy (`cv2.RETR_TREE`) identifies candidate character bounding boxes $b_i = (x_i, y_i, w_i, h_i)$. Extraneous noise and document borders are filtered using relative scale bounds:

$$h_{\min} = 8\text{ px}, \quad h_{\max} = 0.60 \cdot H_{\text{img}}, \quad 0.15 \le \frac{w_i}{h_i} \le 1.80$$

---

### 2. Geometric Spacing & Topological Candidate Ranking

To prevent false groupings across distinct words, candidate bounding box tuples $(b_0, b_1, b_2, b_3)$ must satisfy horizontal pitch consistency and inter-character proximity:

$$|y_i - y_{i+1}| \le \max(3, 0.12 \cdot \bar{h})$$

$$0.50 \cdot \bar{w} \le \Delta x_i \le 1.80 \cdot \bar{h}, \quad \text{where } \Delta x_i = x_{i+1} - x_i$$

$$|\Delta x_1 - \Delta x_2| \le \max(3, 0.20 \cdot \bar{w})$$

Candidate clusters receive topological verification by evaluating internal contours (topological holes $H$):
* $b_0$ (`2`): $H = 0$
* $b_1, b_2$ (`00`): $H \ge 1$

A regional separator bonus is applied if a hyphen or slash is detected adjacent to $b_3$ within $y$-bounds having $h_{\text{sep}} \le 0.65 \cdot \bar{h}$.

---

### 3. Font Family Recognition via Multi-Glyph Shape Correlation

The engine identifies the document's font family by evaluating normalized Intersection-over-Union (IoU) between document character masks and reference rasterizations of candidate typefaces $\mathcal{F} \in \{\text{Open Sans SemiBold}, \text{Inter Regular}, \text{Roboto Regular}\}$:

$$\text{IoU}(C, R) = \frac{\sum_{(x,y)} \left(C(x,y) \land R(x,y)\right)}{\sum_{(x,y)} \left(C(x,y) \lor R(x,y)\right)}$$

$$\text{Score}(\mathcal{F}) = 0.40 \cdot \text{IoU}(C_0, R_2) + 0.30 \cdot \text{IoU}(C_1, R_0) + 0.30 \cdot \text{IoU}(C_2, R_0)$$

$$\hat{\mathcal{F}} = \arg\max_{\mathcal{F}} \text{Score}(\mathcal{F})$$

---

### 4. Baseline Alignment & Grid Reconstruction

To eliminate vertical droop and pitch drift, the target digit bounding box $b_3 = (x_3, y_3, w_3, h_3)$ is calculated using the median metrics of untouched neighbor digits $(b_0, b_1, b_2)$:

$$y_3 = \text{median}\left(y_0, y_1, y_2\right)$$

$$h_3 = \text{median}\left(y_0 + h_0, y_1 + h_1, y_2 + h_2\right) - y_3$$

$$w_3 = \text{median}\left(w_0, w_1, w_2\right)$$

$$\bar{p} = \frac{(x_1 - x_0) + (x_2 - x_1)}{2}, \quad x_3 = \text{round}\left(x_0 + 3 \cdot \bar{p}\right)$$

---

### 5. Point Spread Function (PSF) Measurement & Edge Softness Blending

To match compression artifacts and optical camera blur, the mean edge gradient magnitude $G_{\text{target}}$ is sampled across untouched glyph boundaries using Sobel operators:

$$G(x,y) = \sqrt{\left(\frac{\partial I}{\partial x}\right)^2 + \left(\frac{\partial I}{\partial y}\right)^2}$$

$$G_{\text{target}} = \frac{1}{|\Omega_{\text{edge}}|} \sum_{(x,y) \in \Omega_{\text{edge}}} G(x,y), \quad \Omega_{\text{edge}} = \{(x,y) \mid G(x,y) > 40\}$$

The required continuous Gaussian blur parameter $\sigma_{\text{doc}}$ is derived:

$$\sigma_{\text{doc}} = \begin{cases} 0.0 & \text{if } G_{\text{target}} \ge 580.0 \\ \min\left(1.20, \max\left(0.15, \frac{580.0 - G_{\text{target}}}{220.0}\right)\right) & \text{if } G_{\text{target}} < 580.0 \end{cases}$$

The replacement glyph is rasterized onto a $4\times$ supersampled canvas ($S = 4$), convolved with $\sigma_{\text{canvas}} = \sigma_{\text{doc}} \cdot S$, and reconstructed via Lanczos-3 downsampling. 

Alpha compositing blends the synthetic stroke with the background patch:

$$\mathbf{I}_{\text{out}}(x,y) = \boldsymbol{\alpha}(x,y) \odot \mathbf{C}_{\text{fg}} + \left(1 - \boldsymbol{\alpha}(x,y)\right) \odot \mathbf{I}_{\text{bg}}(x,y)$$

---

## Technology Stack

* **Language**: Python 3.11+
* **Computer Vision**: OpenCV (`opencv-python-headless`), NumPy
* **Typography & Vector Graphics**: Pillow (`PIL.ImageFont`, `PIL.ImageDraw`)
* **API Framework**: FastAPI, Uvicorn, Starlette (GZip Middleware)
* **Frontend**: Vanilla HTML5, CSS3 (JetBrains Mono, custom properties), JavaScript (Clipboard API, Drag & Drop API, Async Fetch)
* **Containerization & Deployment**: Docker, Fly.io

---

## Project Structure

```
webapp/
├── Dockerfile                  # Minimal python:3.11-slim container specification
├── fly.toml                    # Production deployment configuration
├── requirements.txt            # Python dependencies
├── main.py                     # FastAPI application endpoints
├── diditforthelulz.py          # Core computer vision & synthesis engine
├── fonts/                      # TrueType font assets
│   ├── Inter-Regular.ttf
│   ├── Inter-Bold.ttf
│   ├── OpenSans-SemiBold-Static.ttf
│   ├── OpenSans-Bold-Static.ttf
│   ├── Roboto-Regular.ttf
│   └── Roboto-Bold.ttf
└── static/                     # Frontend interface
    ├── index.html
    ├── style.css
    └── script.js
```

---

## Local Development

### 1. Installation

```bash
git clone https://github.com/NilayShenai/nirAADHAR.git
cd nirAADHAR
pip install -r requirements.txt
```

### 2. Running the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Access the interface at `http://localhost:8080`.

---

## API Reference

### `POST /api/modify`

Modifies the target digit in an uploaded document image.

#### Request (Multipart Form Data):
* `file`: Image binary (`image/jpeg`, `image/png`, `image/webp`)
* `digit`: Replacement numeral (`0`–`9`)
* `platform`: Optional (`auto`, `opensans`, `inter`, `roboto`; defaults to `auto`)

#### Response:
* Binary stream (`image/png`) containing the modified document.

---

## License

MIT License.
