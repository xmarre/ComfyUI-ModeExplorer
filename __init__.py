"""ComfyUI discovers this module as a custom node pack."""

if __package__:
    from .mode_explorer.nodes import ModeExplorer, ModeExplorerLoadResponse, ModeExplorerResponse
else:
    # A source checkout with a hyphenated name can be imported as a top-level
    # module by test discovery. ComfyUI loads this entry point as a package.
    from mode_explorer.nodes import ModeExplorer, ModeExplorerLoadResponse, ModeExplorerResponse

NODE_CLASS_MAPPINGS = {
    "ModeExplorerKrea2": ModeExplorer,
    "ModeExplorerResponseKrea2": ModeExplorerResponse,
    "ModeExplorerLoadResponseKrea2": ModeExplorerLoadResponse,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModeExplorerKrea2": "ModeExplorer (Krea 2)",
    "ModeExplorerResponseKrea2": "ModeExplorer Response (Krea 2)",
    "ModeExplorerLoadResponseKrea2": "ModeExplorer Load Response (Krea 2)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
