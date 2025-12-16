"""
Advanced Image Filter with Multiple Artistic Effects
This program applies various artistic filters to images including blur, edge detection,
posterization, and custom color transformations.
"""

from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


def apply_blur_filter(image_path, radius=2, output_path=None):
    """
    Apply Gaussian blur filter.
    
    Args:
        image_path: Path to input image
        radius: Blur radius (default 2)
        output_path: Path to save output
    """
    try:
        img = Image.open(image_path)
        img_blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
        
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_blurred{ext}"
        
        img_blurred.save(output_path)
        print(f"  Blur filter applied. Saved as '{output_path}'")
        return output_path
    except Exception as e:
        print(f"Error applying blur filter: {e}")
        return None


def apply_vintage_filter(image_path, output_path=None):
    """
    Apply vintage/sepia effect.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((256, 256))
        # Convert to sepia tone
        img_array = np.array(img, dtype=np.float32)
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia = np.dot(img_array, sepia_filter.T)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        
        # Reduce saturation for vintage effect
        img_vintage = Image.fromarray(sepia)
        enhancer = ImageEnhance.Color(img_vintage)
        img_vintage = enhancer.enhance(0.7)
        
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_vintage{ext}"
        
        img_vintage.save(output_path)
        print(f"  Vintage filter applied. Saved as '{output_path}'")
        return output_path
    except Exception as e:
        print(f"Error applying vintage filter: {e}")
        return None


def apply_edge_detection_filter(image_path, output_path=None):
    """
    Apply edge detection filter (Sobel-like effect).
    
    Args:
        image_path: Path to input image
        output_path: Path to save output
    """
    try:
        img = Image.open(image_path).convert('L')
        img = img.resize((256, 256))
        # Apply edge detection
        img_edges = img.filter(ImageFilter.FIND_EDGES)
        
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_edges{ext}"
        
        img_edges.save(output_path)
        print(f"  Edge detection filter applied. Saved as '{output_path}'")
        return output_path
    except Exception as e:
        print(f"Error applying edge detection filter: {e}")
        return None


def apply_posterize_filter(image_path, bits=3, output_path=None):
    """
    Apply posterization effect (reduce color palette).
    
    Args:
        image_path: Path to input image
        bits: Number of bits per color channel (lower = more posterized)
        output_path: Path to save output
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((256, 256))
        # Posterize: reduce colors
        img_posterized = ImageOps.posterize(img, bits=bits)
        
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_posterized{ext}"
        
        img_posterized.save(output_path)
        print(f"  Posterize filter applied. Saved as '{output_path}'")
        return output_path
    except Exception as e:
        print(f"Error applying posterize filter: {e}")
        return None


def apply_neon_glow_filter(image_path, output_path=None):
    """
    Apply neon glow effect using edge detection and color manipulation.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((256, 256))
        # Create neon effect using edge detection and color inversion
        img_edges = img.filter(ImageFilter.FIND_EDGES)
        
        # Invert colors for neon look
        img_neon = ImageOps.invert(img_edges.convert('RGB'))
        
        # Boost contrast and saturation
        enhancer = ImageEnhance.Contrast(img_neon)
        img_neon = enhancer.enhance(2.0)
        
        enhancer = ImageEnhance.Color(img_neon)
        img_neon = enhancer.enhance(1.5)
        
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_neon{ext}"
        
        img_neon.save(output_path)
        print(f"  Neon glow filter applied. Saved as '{output_path}'")
        return output_path
    except Exception as e:
        print(f"Error applying neon glow filter: {e}")
        return None


def apply_oil_painting_filter(image_path, radius=4, output_path=None):
    """
    Apply oil painting effect using median filter.
    
    Args:
        image_path: Path to input image
        radius: Radius for median filter effect
        output_path: Path to save output
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((256, 256))
        # Apply median filter for oil painting effect
        img_oil = img.filter(ImageFilter.MedianFilter(size=2*radius+1))
        
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_oil_painting{ext}"
        
        img_oil.save(output_path)
        print(f"  Oil painting filter applied. Saved as '{output_path}'")
        return output_path
    except Exception as e:
        print(f"Error applying oil painting filter: {e}")
        return None


def display_filter_menu():
    """Display available filters."""
    print("\n" + "="*60)
    print("AVAILABLE ARTISTIC FILTERS")
    print("="*60)
    print("1. Blur Filter (classic Gaussian blur)")
    print("2. Vintage/Sepia (nostalgic film effect)")
    print("3. Edge Detection (outline effect)")
    print("4. Posterize (reduce color palette)")
    print("5. Neon Glow (high-contrast edge effect)")
    print("6. Oil Painting (smooth artistic effect)")
    print("0. Exit")
    print("="*60)


def main():
    """Main program loop."""
    print("\n" + "="*60)
    print("ADVANCED IMAGE FILTER PROCESSOR")
    print("="*60)
    
    while True:
        display_filter_menu()
        choice = input("\nSelect a filter (0-6): ").strip()
        
        if choice == "0":
            print("Goodbye!")
            break
        
        image_path = input("Enter image filename: ").strip()
        
        if not Path(image_path).exists():
            print(f"File not found: {image_path}\n")
            continue
        
        print("\nProcessing image...")
        
        if choice == "1":
            apply_blur_filter(image_path, radius=3)
        elif choice == "2":
            apply_vintage_filter(image_path)
        elif choice == "3":
            apply_edge_detection_filter(image_path)
        elif choice == "4":
            apply_posterize_filter(image_path, bits=3)
        elif choice == "5":
            apply_neon_glow_filter(image_path)
        elif choice == "6":
            apply_oil_painting_filter(image_path, radius=4)
        else:
            print("Invalid choice. Please try again.")
        
        print()


if __name__ == "__main__":
    main()
