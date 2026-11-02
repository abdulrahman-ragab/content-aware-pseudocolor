# content-aware-pseudocolor

## Fixed Viridis baseline

The CPU-only baseline uses OpenCV's fixed 256-entry Viridis lookup table. A given
8-bit intensity always maps to the same RGB color, independent of the image.

### Run

Place the downloaded patient sample's images under `data/raw/images` (or directly
under `data/raw`). Keep its CSV under `data/raw/metadata`. PNG discovery is recursive;

```powershell
.\.venv\Scripts\python.exe src\baseline\baseline.py
```

```powershell
.\.venv\Scripts\python.exe src\baseline\baseline.py --input-dir data\raw\images --output-dir data\processed\viridis-sample --gallery-size 10
```

The default output is `data/processed/viridis`:

- `images/`: Viridis PNGs preserving source-relative paths and filenames.
- `comparisons/`: original grayscale versus Viridis for the first 10 sorted images,
  at native resolution, with a labeled 0-255 intensity colorbar.
- `manifest.json`: completed-run record containing input root, relative filenames,
  image sizes, library versions, and fixed processing settings.

### Input formats and reruns

The loader accepts native 8-bit grayscale PNGs and grayscale stored as RGB/RGBA,
but only when all three color channels are identical and any alpha channel is
fully opaque (255). It takes one identical channel without changing intensities
and logs this decoding step. True-color, transparent, 16-bit, and corrupt images
are rejected rather than silently converted. The run manifest records this policy.

Some NIH PNGs have an inconsistent embedded ICC profile, which can cause a libpng
warning. The loader works with decoded pixel values, not color-profile transforms;
acceptance still requires the grayscale checks above. Original files are unchanged.

The output directory must be new or empty. After a failed run, keep the partial
outputs separate and choose a new output folder, for example:

```powershell
.\.venv\Scripts\python.exe src\baseline\baseline.py --output-dir data\processed\viridis-validated
```

Tests use temporary synthetic PNGs and cover grayscale RGB/RGBA decoding,
transparency rejection, LUT consistency, file round trips, and batch processing:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```
