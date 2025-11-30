# Real_time_Hadamard_paper

Block-based compressive sensing using 2D Total Sequency (TS) ordered Hadamard matrix

**Manuscript**  
"Enhanced Real-Time Image Reconstruction via 2D Hadamard Total Sequency Ordering in Compressive Sensing"  
**Authors:** Dr. Mohammad Amiri, Alireza Ghafari  
**Status:** Under review at The Visual Computer (Springer)

This repository contains the exact code that reproduces all results reported in the manuscript (PSNR, runtime) on Set11, Urban100 datasets.

## Method
- Block size: 32×32 (N = 1024)
- Hadamard matrix reordered by 2D Total Sequency (row + column zero-crossings)
- Measurements: top MR proportion of reordered rows
- Reconstruction: direct pseudo-inverse Φ⁺ (training-free, non-iterative)
- Supports grayscale and RGB images

## Requirements
```bash
pip install numpy scipy scikit-image Pillow pandas tqdm
```

## Usage

1. Place test images in a folder  
2. (Optional) Set the parameters if needed:
   ```python
   MR         = 0.1    # ← Change sampling ratio here (e.g., 0.01, 0.04, 0.1, 0.25)
   block_size = 32      # ← Change block size here (must be power of 2: 16, 32, 64, ...)
   ```
3. Update input/output paths:
   ```python
   input_folder  = 'path/to/images/'
   output_folder = 'path/to/reconstructed/'
   csv_file      = 'results.csv'
   ```
4. Run:
   ```bash
   python TS_Hadamard_CS_Reconstruction.py
   ```

**Output**
- Reconstructed images (`reconstructed_*.png`)
- `results.csv` with per-image and average PSNR/SSIM
- Console summary of timing and quality

**Reference results on Set11 (block=32×32)**

   The following values are obtained when running the script on the included `./data/Set11/` dataset and exactly match **Table 2** in the manuscript:

| Sampling Ratio (MR) | Average PSNR (dB) |
|---------------------|-------------------|
| 0.01                | 19.61             |
| 0.04                | 22.21             | 
| 0.10                | 24.38             | 
| 0.25                | 27.68             | 


## Citation
```bibtex
@article{
  title   = {Enhanced Real-Time Image Reconstruction via 2D Hadamard Total Sequency Ordering in Compressive Sensing},
  author  = {Mohammad Amiri and Alireza Ghafari},
  journal = {The Visual Computer},
  year    = {2025},
  note    = {Under review}
}
```

## License
For academic and research use only.










