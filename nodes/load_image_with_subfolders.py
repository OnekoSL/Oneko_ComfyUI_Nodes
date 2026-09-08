import os

import folder_paths
from nodes import LoadImage


class OnekoLoadImageWithSubfolders(LoadImage):
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        for root, dirs, names in os.walk(input_dir):
            dirs[:] = [name for name in dirs if name != "clipspace"]
            files.extend(
                os.path.relpath(os.path.join(root, name), input_dir).replace("\\", "/")
                for name in names
            )
        images = folder_paths.filter_files_content_types(files, ["image"])
        return {"required": {"image": (sorted(images), {"image_upload": True})}}

    CATEGORY = "Oneko/06 Image/Load"
    DESCRIPTION = "Loads images from input subfolders using ComfyUI's image decoder."


NODE_CLASS_MAPPINGS = {
    "OnekoLoadImageWithSubfolders": OnekoLoadImageWithSubfolders,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoLoadImageWithSubfolders": "Load Image with Subfolders (Oneko)",
}
