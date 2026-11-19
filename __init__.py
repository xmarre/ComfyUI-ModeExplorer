"""ComfyUI discovers this module as a custom node pack."""

if __package__:
    from .mode_explorer.nodes import ModeExplorer
else:
    # A source checkout with a hyphenated name can be imported as a top-level
    # module by test discovery. ComfyUI loads this entry point as a package.
    from mode_explorer.nodes import ModeExplorer

NODE_CLASS_MAPPINGS = {"ModeExplorerKrea2": ModeExplorer}
NODE_DISPLAY_NAME_MAPPINGS = {"ModeExplorerKrea2": "ModeExplorer (Krea 2)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
