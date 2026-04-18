# Color Mottle Corrector

`color-mottle-corrector` is a standalone desktop tool for reducing large-scale background color mottle in stretched OSC astrophotography images.

It is intended for cases where residual red/green and blue/yellow background imbalance remains visible after the normal processing pipeline, such as stacking, calibration, gradient removal, and color calibration.

The tool performs a mask-aware correction in Lab color space using normalized blurred chroma fields to neutralize low-frequency background color artifacts while preserving protected image regions.

## Scope and Intended Use

This tool is designed for stretched, pre-final images.

Use it as a last-resort correction step when your normal astrophotography workflow can no longer clean the background adequately.

It is optimized for large-scale, low-frequency background color mottle. It is not intended to fix high-frequency chroma noise or replace proper calibration and background extraction earlier in the pipeline.

![](assets/comparison_m45.jpg)

## Supported Formats

Input and output formats:

- TIFF: 8-bit integer, 16-bit integer, 32-bit float
- FITS: 8-bit integer, 16-bit integer, 32-bit float

## Installation

Install from PyPI:

```bash
pip install color-mottle-corrector
```

## Run

Start the application:

```bash
color-mottle-corrector
```

## Workflow

### 1. Load Source Image

Choose an input TIFF or FITS file and click **Load**.

The original image will appear in the preview area.

### 2. Build the Protection Mask

Adjust the curve and mask controls, then click **Apply mask**.

![](assets/mask_setup.jpg)

Mask meaning:

- **White**: protected region, no correction applied
- **Black**: full correction applied
- **Gray**: partial correction, blended proportionally

Use **Invert output** to invert the mask response.

Use **Mask blur radius** to soften mask transitions and reduce hard correction boundaries.

### 3. Calculate Correction Fields

Select the Gaussian field scale from the available discrete values:

- 4
- 8
- 16
- 32
- 64

Then click **Calculate** to build the `RG_field` and `BY_field`.

![](assets/field_calculation.jpg)

The preview will show the low-frequency correction fields used to neutralize background color imbalance.

In most cases, the default value `32` works well and does not need adjustment.
Tune this only if field geometry is detected incorrectly, for example when color mottle structures are captured too coarsely or too fragmentedly.

### 4. Apply Correction

Set the correction strengths:

- **RG strength (%)**
- **BY strength (%)**

Then click **Preview** to see the corrected image.

![](assets/correction_applied.jpg)

Practical guidance:

- If the result introduces a visible color cast, reduce correction strength or refine the mask first.
- Better masking usually gives more natural results than simply increasing suppression strength.
- Apply conservatively to avoid flattening legitimate faint color variation in the background.

### 5. Save Output

Choose the output path and select the desired format and bit depth.

Then click **Save**.

## Stretch Preview

A global **Stretch image** checkbox is available in the image preview during the Source, Correction, and Save stages.
Enable it to apply a visual stretch for easier inspection of subtle background color mottle.

## Algorithm Development Notebook

This repository also includes [`blotch_detection.ipynb`](blotch_detection.ipynb), used to prototype and validate the core correction algorithm step by step.

## Notes

- This tool is intended for large, low-frequency background color mottle rather than random small-scale color noise.
- It works best on already stretched images where the mottle is clearly visible.
- It should be used conservatively and only when standard processing steps are no longer sufficient.

## Author

Alexey Zaytsev  
lonelylockley@gmail.com

## License

MIT
