# -*- coding: utf-8 -*-
# CellViT Inference Pipeline for Whole Slide Images (WSI) in Memory
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.abspath(current_dir))
sys.path.append(current_dir)
sys.path.append(project_root)

import openslide
from PIL import Image
import numpy as np

class PILSlide:
    def __init__(self, path):
        self.path = path
        self.image = Image.open(path).convert('RGBA')
        self.dimensions = self.image.size
        self.level_count = 1
        self.level_dimensions = (self.dimensions,)
        self.level_downsamples = (1.0,)
        self.properties = {}
        # Add default properties for CellViT (MPP=0.25 -> 40x)
        self.properties["openslide.mpp-x"] = 0.25
        self.properties["openslide.objective-power"] = 40

    def read_region(self, location, level, size):
        # location is (x, y) at level 0
        # level is 0
        # size is (w, h)
        # Note: crop expects (left, top, right, bottom)
        return self.image.crop((location[0], location[1], location[0]+size[0], location[1]+size[1]))

    def get_best_level_for_downsample(self, downsample):
        return 0

    def get_thumbnail(self, size):
        thumb = self.image.resize(size).convert('RGB')
        print(f"DEBUG: PILSlide.get_thumbnail mode={thumb.mode}")
        return thumb

    def close(self):
        self.image.close()

OriginalOpenSlide = openslide.OpenSlide

def OpenSlideWrapper(filename):
    if str(filename).lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
        return PILSlide(filename)
    return OriginalOpenSlide(filename)

openslide.OpenSlide = OpenSlideWrapper

# Force pathopatch to use OpenSlide by hiding cucim
# This ensures it hits the "else" branch and uses our patched OpenSlide
try:
    import pathopatch.utils.tools
    # Store original if needed, or just replace logic since we know we want to hide cucim
    # original_module_exists = pathopatch.utils.tools.module_exists 
    # But checking source, module_exists uses importlib.
    
    def patched_module_exists(name, error="ignore"):
        if name == "cucim":
            return False
        # Fallback to simple import check if original unavailable or complicated
        # Or re-implement basic check
        try:
            import importlib.util
            return importlib.util.find_spec(name) is not None
        except:
            return False

    pathopatch.utils.tools.module_exists = patched_module_exists
except ImportError:
    pass


from cellvit.inference.cli import InferenceWSIParser
from cellvit.inference.inference_memory import CellViTInferenceMemory
from pathlib import Path
import pandas as pd


def main():
    configuration_parser = InferenceWSIParser()
    args = configuration_parser.parse_arguments()
    command = args["command"]

    celldetector = CellViTInferenceMemory(
        model_path=args["model"],
        classifier_path=args["classifier_path"],
        binary=args["binary"],
        gpu=args["gpu"],
        outdir=args["outdir"],
        geojson=args["geojson"],
        graph=args["graph"],
        compression=args["compression"],
        batch_size=args["batch_size"],
        enforce_mixed_precision=args["enforce_amp"],
    )

    if command.lower() == "process_wsi":
        celldetector.logger.info("Processing single WSI file")
        wsi_path = Path(args["wsi_path"])
        wsi_name = wsi_path.stem
        celldetector.process_wsi(
            wsi_path=wsi_path,
            wsi_properties=args.get("wsi_properties", {}),
            resolution=args["resolution"],
        )

    elif command.lower() == "process_dataset":
        celldetector.logger.info("Processing whole dataset")
        if args["filelist"] is not None:
            celldetector.logger.info(f"Loading files from filelist {args['filelist']}")
            wsi_filelist = pd.read_csv(args["filelist"], delimiter=",")
            wsi_filelist = wsi_filelist.to_dict(orient="records")

            for wsi_index, wsi in enumerate(wsi_filelist):
                celldetector.logger.info(f"Progress: {wsi_index+1}/{len(wsi_filelist)}")
                wsi_path = Path(wsi["path"])
                wsi_properties = {}
                if "slide_mpp" in wsi:
                    wsi_properties["slide_mpp"] = wsi["slide_mpp"]
                if "magnification" in wsi:
                    wsi_properties["magnification"] = wsi["magnification"]
                celldetector.process_wsi(
                    wsi_path=wsi_path,
                    wsi_properties=wsi_properties,
                    resolution=args["resolution"],
                )

        elif args["wsi_folder"] is not None:
            celldetector.logger.info(
                f"Loading all files from folder {args['wsi_folder']}. No filelist provided."
            )
            wsi_filelist = [
                f
                for f in sorted(
                    Path(args["wsi_folder"]).glob(f"**/*.{args['wsi_extension']}")
                )
            ]
            for wsi_index, wsi in enumerate(wsi_filelist):
                celldetector.logger.info(f"Progress: {wsi_index+1}/{len(wsi_filelist)}")
                wsi_path = Path(wsi)
                wsi_properties = {}
                # if "slide_mpp" in wsi:
                #     wsi_properties["slide_mpp"] = wsi["slide_mpp"]
                # if "magnification" in wsi:
                #     wsi_properties["magnification"] = wsi["magnification"]
                celldetector.process_wsi(
                    wsi_path=wsi_path,
                    wsi_properties=wsi_properties,
                    resolution=args["resolution"],
                )
        else:
            raise ValueError("Provide either filelist or wsi_folder.")
    celldetector.logger.info("Finished processing")


if __name__ == "__main__":
    main()
