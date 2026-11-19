"""ComfyUI discovers this module as a custom node pack."""

from .mode_explorer.nodes import ModeExplorer

NODE_CLASS_MAPPINGS = {"ModeExplorerKrea2": ModeExplorer}
NODE_DISPLAY_NAME_MAPPINGS = {"ModeExplorerKrea2": "ModeExplorer (Krea 2)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
