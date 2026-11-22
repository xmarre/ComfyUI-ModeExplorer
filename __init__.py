"""ComfyUI discovers this module as a custom node pack."""

if __package__:
    from .mode_explorer.trajectory_review import ModeExplorerTrajectoryReview
    from .mode_explorer.nodes import ModeExplorer, ModeExplorerLoadResponse, ModeExplorerMechanistic, ModeExplorerProjectionGuidance, ModeExplorerResponse
else:
    from mode_explorer.trajectory_review import ModeExplorerTrajectoryReview
    # A source checkout with a hyphenated name can be imported as a top-level
    # module by test discovery. ComfyUI loads this entry point as a package.
    from mode_explorer.nodes import ModeExplorer, ModeExplorerLoadResponse, ModeExplorerMechanistic, ModeExplorerProjectionGuidance, ModeExplorerResponse

NODE_CLASS_MAPPINGS = {
    "ModeExplorerKrea2": ModeExplorer,
    "ModeExplorerResponseKrea2": ModeExplorerResponse,
    "ModeExplorerMechanisticKrea2": ModeExplorerMechanistic,
    "ModeExplorerProjectionGuidanceKrea2": ModeExplorerProjectionGuidance,
    "ModeExplorerLoadResponseKrea2": ModeExplorerLoadResponse,
    "ModeExplorerTrajectoryReviewKrea2": ModeExplorerTrajectoryReview,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModeExplorerKrea2": "ModeExplorer (Krea 2)",
    "ModeExplorerResponseKrea2": "ModeExplorer Response (Krea 2)",
    "ModeExplorerMechanisticKrea2": "ModeExplorer Mechanistic (Krea 2)",
    "ModeExplorerProjectionGuidanceKrea2": "ModeExplorer Projection Guidance (Krea 2)",
    "ModeExplorerLoadResponseKrea2": "ModeExplorer Load Response (Krea 2)",
    "ModeExplorerTrajectoryReviewKrea2": "ModeExplorer Trajectory Review (Krea 2)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
