"""ComfyUI discovers this module as a custom node pack."""

if __package__:
    from .mode_explorer.research_report import ModeExplorerResearchReport
    from .mode_explorer.trajectory_review import ModeExplorerTrajectoryReview
    from .mode_explorer.nodes import ModeExplorer, ModeExplorerLoadResponse, ModeExplorerMechanistic, ModeExplorerProjectionGuidance, ModeExplorerResponse
    from .mode_explorer.rediverse_nodes import (
        ModeExplorerActivationProbe, ModeExplorerReDiverse, ModeExplorerReDiverseCalibrate, ModeExplorerReDiverseDirections,
    )
else:
    from mode_explorer.research_report import ModeExplorerResearchReport
    from mode_explorer.trajectory_review import ModeExplorerTrajectoryReview
    # A source checkout with a hyphenated name can be imported as a top-level
    # module by test discovery. ComfyUI loads this entry point as a package.
    from mode_explorer.nodes import ModeExplorer, ModeExplorerLoadResponse, ModeExplorerMechanistic, ModeExplorerProjectionGuidance, ModeExplorerResponse
    from mode_explorer.rediverse_nodes import (
        ModeExplorerActivationProbe, ModeExplorerReDiverse, ModeExplorerReDiverseCalibrate, ModeExplorerReDiverseDirections,
    )

NODE_CLASS_MAPPINGS = {
    "ModeExplorerKrea2": ModeExplorer,
    "ModeExplorerResponseKrea2": ModeExplorerResponse,
    "ModeExplorerMechanisticKrea2": ModeExplorerMechanistic,
    "ModeExplorerProjectionGuidanceKrea2": ModeExplorerProjectionGuidance,
    "ModeExplorerLoadResponseKrea2": ModeExplorerLoadResponse,
    "ModeExplorerTrajectoryReviewKrea2": ModeExplorerTrajectoryReview,
    "ModeExplorerResearchReportKrea2": ModeExplorerResearchReport,
    "ModeExplorerReDiverseKrea2": ModeExplorerReDiverse,
    "ModeExplorerReDiverseCalibrateKrea2": ModeExplorerReDiverseCalibrate,
    "ModeExplorerReDiverseDirectionsKrea2": ModeExplorerReDiverseDirections,
    "ModeExplorerActivationProbeKrea2": ModeExplorerActivationProbe,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModeExplorerKrea2": "ModeExplorer (Krea 2)",
    "ModeExplorerResponseKrea2": "ModeExplorer Response (Krea 2)",
    "ModeExplorerMechanisticKrea2": "ModeExplorer Mechanistic (Krea 2)",
    "ModeExplorerProjectionGuidanceKrea2": "ModeExplorer Projection Guidance (Krea 2)",
    "ModeExplorerLoadResponseKrea2": "ModeExplorer Load Response (Krea 2)",
    "ModeExplorerTrajectoryReviewKrea2": "ModeExplorer Trajectory Review (Krea 2)",
    "ModeExplorerResearchReportKrea2": "ModeExplorer Research Report (Krea 2)",
    "ModeExplorerReDiverseKrea2": "ModeExplorer ReDiverse (Krea 2)",
    "ModeExplorerReDiverseCalibrateKrea2": "ModeExplorer ReDiverse Calibrate (Krea 2)",
    "ModeExplorerReDiverseDirectionsKrea2": "ModeExplorer ReDiverse Directions (Krea 2)",
    "ModeExplorerActivationProbeKrea2": "ModeExplorer Activation Probe (Krea 2)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
