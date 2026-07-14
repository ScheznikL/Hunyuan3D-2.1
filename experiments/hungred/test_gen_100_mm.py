import argparse
import trimesh
import numpy as np
import json
import zipfile
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import tempfile


def extract_dimensions_from_prompt(prompt: str) -> Dict[str, float]:
    """
    Extract all dimensional values from the prompt text.
    Automatically detects units (mm or cm) and returns values in MILLIMETERS.
    """
    dimensions = {}
    
    # Patterns for millimeters
    patterns_mm = {
        'total_length': r'Total length:\s*([\d.]+)\s*mm',
        'top_width': r'top internal width\s*([\d.]+)\s*mm',
        'bottom_width': r'bottom internal width\s*([\d.]+)\s*mm',
        'channel_depth': r'channel depth\s*([\d.]+)\s*mm',
        'wall_thickness': r'wall thickness is\s*([\d.]+)\s*mm',
        'notch_width': r'notch:\s*([\d.]+)\s*mm wide',
        'notch_depth': r'(\d+\.\d+)\s*mm deep'
    }
    
    # Patterns for centimeters
    patterns_cm = {
        'total_length': r'Total length:\s*([\d.]+)\s*cm',
        'top_width': r'top internal width\s*([\d.]+)\s*cm',
        'bottom_width': r'bottom internal width\s*([\d.]+)\s*cm',
        'channel_depth': r'channel depth\s*([\d.]+)\s*cm',
        'wall_thickness': r'wall thickness is\s*([\d.]+)\s*cm',
        'notch_width': r'notch:\s*([\d.]+)\s*cm wide',
        'notch_depth': r'(\d+\.\d+)\s*cm deep'
    }
    
    # Try mm first
    units_used = None
    for key, pattern in patterns_mm.items():
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            dimensions[key] = float(match.group(1))  # Already in mm
            units_used = 'mm'
    
    # If no mm found, try cm and convert
    if not dimensions:
        for key, pattern in patterns_cm.items():
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                dimensions[key] = float(match.group(1)) * 10  # Convert cm to mm
                units_used = 'cm'
    
    dimensions['_units_detected'] = units_used
    return dimensions


def measure_model_dimensions(mesh_path: str) -> Dict[str, float]:
    """
    Measure actual dimensions from the 3D model.
    Returns measurements in MILLIMETERS (assuming 1 unit = 1 meter).
    """
    try:
        loaded = trimesh.load(mesh_path)
        
        # Handle Scene objects
        if isinstance(loaded, trimesh.Scene):
            if len(loaded.geometry) == 0:
                return {}
            mesh = loaded.dump(concatenate=True)
        else:
            mesh = loaded
        
        # Basic extents (convert to mm: meters * 1000)
        extents = mesh.extents * 1000  # X, Y, Z in mm
        
        # Get bounds
        min_y = mesh.bounds[0][1]
        max_y = mesh.bounds[1][1]
        height = (max_y - min_y) * 1000  # Total height in mm
        
        measurements = {
            'total_length': height,  # Y axis as height/length
            'width_x': extents[0],
            'depth_z': extents[2]
        }
        
        # Measure top width (top 2% slice)
        try:
            top_slice = mesh.section(
                plane_origin=[0, max_y - ((max_y - min_y) * 0.02), 0],
                plane_normal=[0, 1, 0]
            )
            if top_slice:
                measurements['top_width'] = top_slice.extents[0] * 1000  # mm
        except:
            measurements['top_width'] = None
        
        # Measure bottom width (bottom 2% slice)
        try:
            bottom_slice = mesh.section(
                plane_origin=[0, min_y + ((max_y - min_y) * 0.02), 0],
                plane_normal=[0, 1, 0]
            )
            if bottom_slice:
                measurements['bottom_width'] = bottom_slice.extents[0] * 1000  # mm
        except:
            measurements['bottom_width'] = None
        
        # Measure channel depth (approximate as Z extent)
        measurements['channel_depth'] = extents[2]
        
        return measurements
        
    except Exception as e:
        print(f"Error measuring model {mesh_path}: {e}")
        return {}


def calculate_differences(prompt_dims: Dict[str, float], 
                         model_dims: Dict[str, float]) -> Dict[str, Dict]:
    """
    Calculate differences between prompt specifications and model measurements.
    Both values should be in mm.
    Returns absolute difference, percentage difference, and both values.
    """
    differences = {}
    
    # Map prompt dimensions to model measurements
    dimension_mapping = {
        'total_length': 'total_length',
        'top_width': 'top_width',
        'bottom_width': 'bottom_width',
        'channel_depth': 'channel_depth'
    }
    
    for prompt_key, model_key in dimension_mapping.items():
        if prompt_key in prompt_dims and model_key in model_dims:
            if model_dims[model_key] is not None:
                prompt_val = prompt_dims[prompt_key]
                model_val = model_dims[model_key]
                
                abs_diff = abs(prompt_val - model_val)
                pct_diff = (abs_diff / prompt_val) * 100 if prompt_val != 0 else 0
                
                differences[prompt_key] = {
                    'prompt_value': prompt_val,
                    'model_value': model_val,
                    'absolute_diff': abs_diff,
                    'percentage_diff': pct_diff
                }
    
    return differences


def process_all_models(zip_path: str) -> Tuple[List[Dict], Dict[str, List[float]]]:
    """
    Process all models in the zip file.
    Returns list of individual results and aggregated statistics.
    """
    all_results = []
    all_differences = {
        'total_length': [],
        'top_width': [],
        'bottom_width': [],
        'channel_depth': []
    }
    
    units_detected = None
    
    # Create temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Extracting zip file to temporary directory...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Search for 100_test_models directory recursively
        models_dir = None
        for root, dirs, files in os.walk(temp_dir):
            if '100_test_models' in dirs:
                models_dir = Path(root) / '100_test_models'
                break
        
        if models_dir is None or not models_dir.exists():
            print(f"Models directory not found in zip file!")
            print(f"Searched in: {temp_dir}")
            print(f"Available directories:")
            for root, dirs, files in os.walk(temp_dir):
                for d in dirs:
                    print(f"  - {os.path.join(root, d)}")
            return [], {}
        
        print(f"Processing models from: {models_dir}")
        
        # Get all unique UUIDs
        glb_files = list(models_dir.glob('*.glb'))
        print(f"Found {len(glb_files)} GLB files")
        
        for idx, glb_file in enumerate(glb_files, 1):
            uuid = glb_file.stem
            meta_file = models_dir / f"{uuid}_meta.json"
            
            print(f"\n[{idx}/{len(glb_files)}] Processing {uuid}...")
            
            # Load metadata
            if not meta_file.exists():
                print(f"  ⚠️  Metadata file not found: {meta_file}")
                continue
            
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    prompt = metadata.get('prompt', '')
                
                if not prompt:
                    print(f"  ⚠️  No prompt found in metadata")
                    continue
                
                # Extract dimensions from prompt (auto-detects mm or cm)
                prompt_dims = extract_dimensions_from_prompt(prompt)
                detected_units = prompt_dims.pop('_units_detected', None)
                
                if detected_units and units_detected is None:
                    units_detected = detected_units
                    print(f"  📏 Detected units in prompts: {detected_units}")
                
                print(f"  Prompt dimensions: {len(prompt_dims)} found")
                
                # Measure model (returns mm)
                model_dims = measure_model_dimensions(str(glb_file))
                print(f"  Model measurements: {len(model_dims)} taken")
                
                # Calculate differences (both in mm now)
                differences = calculate_differences(prompt_dims, model_dims)
                
                # Store results
                result = {
                    'uuid': uuid,
                    'prompt': prompt,
                    'prompt_dimensions': prompt_dims,
                    'model_dimensions': model_dims,
                    'differences': differences,
                    'units': 'mm'
                }
                all_results.append(result)
                
                # Aggregate differences
                for dim_name, diff_data in differences.items():
                    if dim_name in all_differences:
                        all_differences[dim_name].append(diff_data['absolute_diff'])
                
                # Print summary for this model
                if differences:
                    print(f"  Differences calculated (in mm):")
                    for dim_name, diff_data in differences.items():
                        print(f"    {dim_name}: {diff_data['absolute_diff']:.2f} mm "
                              f"({diff_data['percentage_diff']:.2f}%) - "
                              f"Prompt: {diff_data['prompt_value']:.2f} mm, "
                              f"Model: {diff_data['model_value']:.2f} mm")
                else:
                    print(f"  ⚠️  No matching dimensions found")
                
            except Exception as e:
                print(f"  ❌ Error processing {uuid}: {e}")
                continue
    
    print(f"\n📏 All prompts used units: {units_detected}")
    return all_results, all_differences


def calculate_statistics(all_differences: Dict[str, List[float]]) -> Dict:
    """
    Calculate average, median, std dev, min, max for each dimension.
    Values in mm.
    """
    stats = {}
    
    for dim_name, diff_list in all_differences.items():
        if diff_list:
            stats[dim_name] = {
                'count': len(diff_list),
                'mean': np.mean(diff_list),
                'median': np.median(diff_list),
                'std': np.std(diff_list),
                'min': np.min(diff_list),
                'max': np.max(diff_list)
            }
        else:
            stats[dim_name] = {
                'count': 0,
                'mean': 0,
                'median': 0,
                'std': 0,
                'min': 0,
                'max': 0
            }
    
    return stats


def save_results(all_results: List[Dict], stats: Dict, output_dir: str):
    """
    Save detailed results and summary statistics to JSON files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    detailed_file = os.path.join(output_dir, 'detailed_results.json')
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Detailed results saved to: {detailed_file}")
    
    # Save statistics
    stats_file = os.path.join(output_dir, 'statistics_summary.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Statistics summary saved to: {stats_file}")
    
    # Create human-readable summary
    summary_file = os.path.join(output_dir, 'summary_report.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BATCH MODEL ANALYSIS - SUMMARY REPORT (MILLIMETERS)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total models analyzed: {len(all_results)}\n")
        f.write(f"Units: millimeters (mm)\n\n")
        
        f.write("AVERAGE DIFFERENCES BETWEEN PROMPT AND GENERATED MODELS\n")
        f.write("-" * 80 + "\n\n")
        
        for dim_name, stat_data in stats.items():
            f.write(f"{dim_name.upper().replace('_', ' ')}:\n")
            f.write(f"  Number of measurements: {stat_data['count']}\n")
            f.write(f"  Average difference: {stat_data['mean']:.3f} mm ({stat_data['mean']/10:.3f} cm)\n")
            f.write(f"  Median difference: {stat_data['median']:.3f} mm ({stat_data['median']/10:.3f} cm)\n")
            f.write(f"  Standard deviation: {stat_data['std']:.3f} mm ({stat_data['std']/10:.3f} cm)\n")
            f.write(f"  Min difference: {stat_data['min']:.3f} mm ({stat_data['min']/10:.3f} cm)\n")
            f.write(f"  Max difference: {stat_data['max']:.3f} mm ({stat_data['max']/10:.3f} cm)\n")
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("OVERALL AVERAGE ACROSS ALL DIMENSIONS\n")
        f.write("=" * 80 + "\n\n")
        
        all_means = [stat_data['mean'] for stat_data in stats.values() if stat_data['count'] > 0]
        if all_means:
            overall_mean = np.mean(all_means)
            f.write(f"Overall average difference: {overall_mean:.3f} mm ({overall_mean/10:.3f} cm)\n")
    
    print(f"✅ Summary report saved to: {summary_file}")


def main(zip_path, output_dir):
    # Configuration - UPDATE THESE PATHS
    #  = '/dcs/large/u5745134/batch_100/batch_CKPT00030000_100_res_MM_PROMPTS.zip'
    #  = '/home/claude/analysis_results_mm'
    
    print("=" * 80)
    print("BATCH MODEL ANALYSIS - Starting Processing (MM UNITS)")
    print("=" * 80)
    print(f"Zip file: {zip_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Process all models
    all_results, all_differences = process_all_models(zip_path)
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("CALCULATING STATISTICS")
    print("=" * 80)
    stats = calculate_statistics(all_differences)
    
    # Print statistics to console
    print("\nSTATISTICS SUMMARY (MILLIMETERS):")
    print("-" * 80)
    for dim_name, stat_data in stats.items():
        print(f"\n{dim_name.upper().replace('_', ' ')}:")
        print(f"  Count: {stat_data['count']}")
        print(f"  Average difference: {stat_data['mean']:.3f} mm ({stat_data['mean']/10:.2f} cm)")
        print(f"  Median difference: {stat_data['median']:.3f} mm ({stat_data['median']/10:.2f} cm)")
        print(f"  Std deviation: {stat_data['std']:.3f} mm ({stat_data['std']/10:.2f} cm)")
        print(f"  Range: {stat_data['min']:.3f} - {stat_data['max']:.3f} mm")
    
    # Calculate overall average
    all_means = [stat_data['mean'] for stat_data in stats.values() if stat_data['count'] > 0]
    if all_means:
        overall_mean = np.mean(all_means)
        print(f"\n{'=' * 80}")
        print(f"OVERALL AVERAGE DIFFERENCE: {overall_mean:.3f} mm ({overall_mean/10:.2f} cm)")
        print(f"{'=' * 80}")
    
    # Save results
    save_results(all_results, stats, output_dir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", required=True, type=str,)
    parser.add_argument("--output_metrics", required=True, type=str) 
    parser.add_argument("--base_dir", required=True, type=str) 
    parser.add_argument("--batch_name", required=True, type=str) 
    return parser.parse_args()
    

if __name__ == "__main__":
    
    args = get_args()
    main(args.zip_path, args.output_metrics)