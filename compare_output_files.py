import os
from pathlib import Path

base_dir = Path(r"c:\Users\samar\Documents\CVPR\Interpretation\CellVit\CellViT-plus-plus-main\results\512_final_dataset")
images_dir = base_dir / "images"
geojsons_dir = base_dir / "geojsons"

def get_basenames(directory, extension):
    return {f.stem for f in directory.glob(f"*{extension}")}

image_basenames = get_basenames(images_dir, ".png")
geojson_basenames = get_basenames(geojsons_dir, ".geojson")

print(f"Total Images: {len(image_basenames)}")
print(f"Total GeoJSONs: {len(geojson_basenames)}")

only_in_images = image_basenames - geojson_basenames
only_in_geojsons = geojson_basenames - image_basenames

print(f"In Images but not GeoJSONs: {len(only_in_images)}")
print(f"In GeoJSONs but not Images: {len(only_in_geojsons)}")

if only_in_images:
    print("\nExamples in Images but not in GeoJSONs:")
    for i, name in enumerate(list(only_in_images)[:5]):
        print(f"  {name}")

if only_in_geojsons:
    print(f"\nDeleting {len(only_in_geojsons)} orphaned GeoJSONs...")
    for name in only_in_geojsons:
        file_path = geojsons_dir / f"{name}.geojson"
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path.name}")
        except Exception as e:
            print(f"Error deleting {file_path.name}: {e}")
            
    print("Cleanup complete.")
