# Image Classification Project - Setup & Execution Guide

## Quick Start

This guide helps you set up and run the Image Classification project with Grad-CAM visualization and artistic filters.

## Installation Steps

### Step 1: Verify Python Installation

Check if Python 3.8+ is installed:

```powershell
python --version
```

**If not installed**, download from: https://www.python.org/downloads/
- During installation, **check "Add Python to PATH"**

### Step 2: Navigate to Project Directory

```powershell
cd "c:\Artificial_Intelligence_Ecosystem\Artificial_Intelligence_Ecosystem\Image_Classification"
```

### Step 3: Create Virtual Environment

```powershell
python -m venv venv
```

### Step 4: Activate Virtual Environment

```powershell
# Windows
venv\Scripts\Activate.ps1

# If you get an execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 5: Upgrade pip and Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: First installation may take 5-10 minutes as TensorFlow/Keras download and install.

## Running the Programs

### Option A: Image Classification with Grad-CAM

```powershell
python base_classifier_with_gradcam.py
```

**Usage**:
1. Program displays title screen
2. Enter image filename when prompted
   - Example: `dog.jpg`
   - Or full path: `C:\Users\YourName\Pictures\dog.jpg`
3. Program outputs:
   - Top 3 predictions with confidence scores
   - Grad-CAM heatmap visualization
   - File saved as `imagename_gradcam.png`
4. Type 'exit' to quit

### Option B: Advanced Image Filters

```powershell
python advanced_filter.py
```

**Usage**:
1. Program displays filter menu
2. Select a filter (1-6):
   - 1 = Blur
   - 2 = Vintage/Sepia
   - 3 = Edge Detection
   - 4 = Posterize
   - 5 = Neon Glow
   - 6 = Oil Painting
3. Enter image filename
4. Program applies filter and saves result
5. Choose another filter or exit (0)

## Project Files

### Programs

| File | Purpose |
|------|---------|
| `base_classifier.py` | Original classifier (reference) |
| `base_classifier_with_gradcam.py` | **Main classifier - use this** |
| `basic_filter.py` | Original blur filter (reference) |
| `advanced_filter.py` | **Main filter program - use this** |

### Documentation

| File | Purpose |
|------|---------|
| `CODE_EXPLANATIONS.md` | Line-by-line code explanations |
| `FINAL_REPORT_TEMPLATE.md` | Report template with sections to fill |
| `SETUP_AND_EXECUTION.md` | **This file** |
| `requirements.txt` | Python dependencies |

## Understanding the Code

### base_classifier_with_gradcam.py

**What it does**:
1. Loads your image
2. Resizes to 224x224 pixels (required by MobileNetV2)
3. Preprocesses pixels (normalizes values)
4. Passes through neural network
5. Gets top 3 predictions
6. Generates Grad-CAM heatmap (shows which parts of image model focused on)
7. Overlays heatmap on original image
8. Saves result as `imagename_gradcam.png`

**Key Functions**:
- `make_gradcam_heatmap()` - Generates the attention visualization
- `save_and_display_gradcam()` - Creates overlay image
- `classify_image_with_gradcam()` - Main classification pipeline

### advanced_filter.py

**What it does**:
1. Loads your image
2. Applies selected filter
3. Saves result with filter name in filename

**Available Filters**:
1. **Blur** - Smooth, dreamy effect (Gaussian blur with radius=3)
2. **Vintage** - Nostalgic sepia tone with reduced saturation
3. **Edge Detection** - Line drawing showing object outlines
4. **Posterize** - Bold color blocks (3-bit reduction)
5. **Neon Glow** - Bright glowing edges (inverted edge detection + contrast)
6. **Oil Painting** - Smooth, painterly appearance

## Creating Your Report

Use `FINAL_REPORT_TEMPLATE.md` as a template:

1. Copy the template
2. Replace [bracketed sections] with your findings
3. Include:
   - Test images and results
   - Grad-CAM observations
   - Filter effects and comparisons
   - Your reflections on AI-assisted development

### What to Include in Report

**Part 1: Image Classification**
- Describe 3 test images
- Show top predictions and confidence scores
- Analyze Grad-CAM heatmaps (what did model focus on?)
- Did the model make correct predictions? Why/why not?

**Part 2: Artistic Filters**
- Show before/after images for each filter
- Describe the visual effect of each
- Which filters worked best and why?
- What real-world applications would each filter have?

**Part 3: AI Assistance**
- What prompts worked well?
- What challenges did you face?
- How did AI help your learning?
- What would you do differently?

## Troubleshooting

### Python Not Found

**Error**: "Python was not found"

**Solution**: 
1. Install Python from https://www.python.org/downloads/
2. Make sure "Add Python to PATH" is checked
3. Restart PowerShell and try again

### TensorFlow Not Installing

**Error**: "Failed to build tensorflow" or similar

**Alternatives**:
```powershell
# Try CPU-only version (smaller, faster to install)
pip install tensorflow-cpu

# Or use pre-built wheel
pip install --no-cache-dir tensorflow
```

### Module Import Errors

**Error**: "ModuleNotFoundError: No module named 'tensorflow'"

**Solution**:
1. Verify virtual environment is activated
2. Check activation: prompt should show `(venv)`
3. Reinstall: `pip install tensorflow`

### Memory Issues (TensorFlow too large)

**Error**: "MemoryError" or installation takes very long

**Solution**:
1. Ensure you have at least 5GB free disk space
2. Close other applications
3. Try: `pip install --no-cache-dir -r requirements.txt`

### Image Not Found

**Error**: "File not found: imagename.jpg"

**Solution**:
1. Place image in same folder as Python script, OR
2. Use full path: `C:\Users\YourName\Pictures\image.jpg`
3. Check file extension is correct (.jpg, .png, etc.)

### Grad-CAM Shows Blank Heatmap

**Possible causes**:
- Image loading failed
- Image format not supported
- Model having trouble with image content

**Solutions**:
1. Try different image file
2. Try different format (JPG instead of PNG)
3. Check image displays correctly on Windows

## Advanced Tips

### Using GPU for Faster Processing

If you have NVIDIA GPU with CUDA support:

```powershell
pip uninstall tensorflow tensorflow-cpu
pip install tensorflow-gpu
```

This will speed up processing 10-100x.

### Processing Multiple Images

Modify programs to loop through image directory:

```python
from pathlib import Path

for image_file in Path('.').glob('*.jpg'):
    classify_image_with_gradcam(str(image_file))
```

### Adjusting Filter Parameters

Edit `advanced_filter.py` to change filter strengths:

```python
# For blur (increase radius for more blur)
apply_blur_filter(image_path, radius=5)

# For posterize (lower bits = more posterized)
apply_posterize_filter(image_path, bits=2)
```

### Combining Filters

Create a new filter by chaining multiple effects:

```python
def apply_combined_filter(image_path):
    img = Image.open(image_path)
    img = apply_blur_first(img)
    img = apply_vintage_effect(img)
    return img
```

## Next Steps After Setup

1. **Test with Sample Images**
   - Use simple images (objects, faces, animals)
   - Avoid heavily cluttered images initially

2. **Understand the Code**
   - Read CODE_EXPLANATIONS.md
   - Run code with different images
   - Modify parameters and observe effects

3. **Document Results**
   - Save 3+ test images and results
   - Screenshot outputs
   - Note observations

4. **Write Report**
   - Use FINAL_REPORT_TEMPLATE.md
   - Include images and analysis
   - Reflect on learning process

5. **Submit to GitHub**
   - Create folder with code and outputs
   - Add report document
   - Push to your forked repository

## File Locations Summary

```
Image_Classification/
├── Programs (Run these):
│   ├── base_classifier_with_gradcam.py  ← Use this for classification
│   └── advanced_filter.py                ← Use this for filters
│
├── Documentation (Read these):
│   ├── CODE_EXPLANATIONS.md
│   ├── FINAL_REPORT_TEMPLATE.md
│   └── SETUP_AND_EXECUTION.md (this file)
│
└── Generated Files (Created by programs):
    ├── image1_gradcam.png
    ├── image2_vintage.png
    └── ...
```

## Quick Reference Commands

```powershell
# Activate environment
venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt

# Run classifier
python base_classifier_with_gradcam.py

# Run filters
python advanced_filter.py

# List installed packages
pip list

# Deactivate environment
deactivate
```

## Getting Help

1. **Code Questions** → See CODE_EXPLANATIONS.md
2. **Algorithm Questions** → See FINAL_REPORT_TEMPLATE.md appendices
3. **Setup Issues** → See Troubleshooting section above
4. **Python/TensorFlow** → Official documentation online
5. **This Project** → Review inline code comments

## Success Indicators

✅ You're ready when:
- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All packages installed successfully
- [ ] base_classifier_with_gradcam.py runs without errors
- [ ] advanced_filter.py runs without errors
- [ ] Can classify a test image and see predictions
- [ ] Grad-CAM heatmap generates and saves

## Next Checkpoint

Once setup is complete:
1. Classify 3 images with Grad-CAM
2. Apply 3-4 different filters to test images
3. Save all results
4. Begin writing report

---

**Questions?** Check CODE_EXPLANATIONS.md or review code comments in Python files.
