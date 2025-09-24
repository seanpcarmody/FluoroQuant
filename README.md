HEAD
# FluoroQuant v2.0

A comprehensive multi-channel fluorescence microscopy analysis platform for quantitative cellular imaging and colocalization studies.

![FluoroQuant Interface](screenshot.png)

## Features

- **Multi-channel Image Processing**: Load and analyze up to 3 fluorescence channels simultaneously
- **Advanced Thresholding**: Multiple algorithms including Otsu, Triangle, Li, Yen, and adaptive local thresholding
- **Colocalization Analysis**: Manders coefficients, Pearson correlation, and distance-based metrics
- **Batch Processing**: Intelligent pattern detection for high-throughput analysis
- **Interactive GUI**: Real-time parameter adjustment with live preview
- **Comprehensive Export**: CSV, JSON, Excel, and image overlay exports

## Installation

### Requirements

- Python 3.7+
- Required packages (install via pip):

```bash
pip install -r requirements.txt
```

### Quick Start

1. Clone the repository:

```bash
git clone https://github.com/seanpcarmody/fluoroquant.git
cd fluoroquant
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python FluroMain.py
```

## Usage

### Single Image Analysis

1. **Load Images**: Click "Load Image" for each channel (supports TIFF, PNG, JPG)
2. **Adjust Parameters**: Use Quick or Advanced tabs to optimize processing
3. **Process**: Click "Process" or enable auto-update for real-time results
4. **Export Results**: Choose formats in Export tab

### Batch Processing

1. **Learn Pattern**: Load 2+ channels to establish naming pattern
2. **Select Folder**: Choose directory containing matching image sets
3. **Process Batch**: Review detected groups and start batch analysis
4. **Auto-Export**: Results automatically saved to batch folder

## Key Algorithms

### Colocalization Analysis

- **Manders Coefficients**: Quantifies spatial overlap between channels
- **Pearson Correlation**: Measures intensity correlation in overlapping regions
- **Distance Analysis**: Calculates nearest-neighbor distances between objects

### Image Processing Pipeline

1. **Preprocessing**: Gaussian smoothing, median filtering, CLAHE enhancement
2. **Thresholding**: Adaptive algorithms for signal/background separation  
3. **Object Detection**: Connected component labeling with size filtering
4. **Feature Extraction**: Morphological and intensity measurements

## Architecture

```
FluoroQuant/
├── FluroMain.py          # Main application controller
├── fluoro_processor.py   # Image processing algorithms
├── fluoro_analyzer.py    # Quantitative analysis methods
├── fluoro_gui.py         # User interface components
├── fluoro_batch.py       # Batch processing logic
├── fluoro_export.py      # Data export utilities
└── requirements.txt      # Python dependencies
```

## Example Output

**Single Channel Results:**

- Object count, area measurements, fluorescence intensity
- Coverage percentage, signal-to-background ratio
- Morphological parameters (circularity, aspect ratio)

**Multi-Channel Analysis:**

- Colocalization coefficients (M1, M2)
- Spatial correlation metrics
- Distance distributions between channel objects

## Technical Details

### Performance Optimizations

- Vectorized NumPy operations for image processing
- Scipy spatial algorithms for distance calculations
- Threading for non-blocking batch processing

### Supported Formats

- **Input**: TIFF, PNG, JPG, BMP (8-bit and 16-bit)
- **Output**: CSV, JSON, Excel, PNG overlays

## Development

Built with:

- **GUI**: Tkinter with custom dark theme
- **Image Processing**: OpenCV, scikit-image
- **Analysis**: NumPy, SciPy, pandas
- **Visualization**: Matplotlib

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

MIT License - see LICENSE file for details

## Citation

If you use FluoroQuant in your research, please cite:

```
FluoroQuant: Multi-Channel Fluorescence Analysis Platform
Sean Carmody, 2025
https://github.com/seanpcarmody/fluoroquant
```
=======
# FluoroQuant
Dynamic image analysis program built for multi-channel microscopy images. Capable of both batch processing and individual image processing with live and integrated image transformation previews. Provides quantitative analyses for selected microscopy channels.

