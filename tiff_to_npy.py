#!/usr/bin/env python3
"""
Convert downloaded GeoTIFF embeddings to numpy format for faster loading.

This script converts:
  austin_YYYY_800m.tif → austin_embeddings_YYYY_800m_64d.npy

The numpy format loads much faster during training.
"""

import rasterio
import numpy as np
import os
from glob import glob

def convert_tif_to_npy(tif_file, output_dir=None):
    """Convert a GeoTIFF embedding file to numpy format"""
    
    basename = os.path.basename(tif_file)
    year = basename.split('_')[1]  # Extract year from austin_2023_800m.tif
    
    if output_dir is None:
        output_dir = os.path.dirname(tif_file)
    
    # Output filename
    output_file = os.path.join(output_dir, f"austin_embeddings_{year}_800m_64d.npy")
    
    # Check if already converted
    if os.path.exists(output_file):
        print(f"Already exists: {output_file}")
        return output_file
    
    print(f"Converting {basename}...")
    
    # Read all bands from TIF
    with rasterio.open(tif_file) as src:
        n_bands = src.count
        height = src.height
        width = src.width
        
        print(f"  Bands: {n_bands}, Shape: [{height}, {width}]")
        
        # Read all bands
        data = src.read()  # Returns [bands, height, width]
        
        # Handle NaN values (replace with 0)
        data = np.nan_to_num(data, nan=0.0)
        
        # Check data
        print(f"  Value range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"  Mean: {data.mean():.3f}, Std: {data.std():.3f}")
    
    # Save as numpy
    np.save(output_file, data)
    print(f"  ✓ Saved to: {output_file}")
    print(f"  Size: {os.path.getsize(output_file) / 1024**2:.1f} MB")
    
    return output_file

def main():
    input_dir = "/scratch/09295/naveens/downscale_latent_ensemble/google_embeddings_austin"
    
    print("="*80)
    print("CONVERTING GEOTIFF EMBEDDINGS TO NUMPY FORMAT")
    print("="*80)
    print()
    
    # Find all TIF files
    tif_files = sorted(glob(os.path.join(input_dir, "austin_*_800m.tif")))
    
    if not tif_files:
        print(f"No TIF files found in {input_dir}")
        return
    
    print(f"Found {len(tif_files)} TIF files")
    print()
    
    # Convert each file
    converted_files = []
    for tif_file in tif_files:
        try:
            output_file = convert_tif_to_npy(tif_file)
            converted_files.append(output_file)
            print()
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()
    
    # Summary
    print("="*80)
    print("CONVERSION COMPLETE")
    print("="*80)
    print(f"Converted {len(converted_files)} files")
    print()
    print("Files created:")
    for f in converted_files:
        size = os.path.getsize(f) / 1024**2
        print(f"  {os.path.basename(f)} ({size:.1f} MB)")
    
    print()
    print("These files will be automatically loaded by the training system!")

if __name__ == "__main__":
    main()
