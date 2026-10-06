import argparse
import json
import logging
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress upstream PyTorch deprecation warnings before loading the pipeline
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=r".*torch\.jit\.script.*is deprecated.*"
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=r".*torch\.cuda\.amp\.custom_fwd.*"
)

from config import config  # noqa: E402
from sfm_core.exporter import export_point_cloud_ply  # noqa: E402
from sfm_core.incremental_engine import IncrementalSfMEngine  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """
    Constructs and returns the Command Line Interface (CLI) argument parser.
    Defines required parameters (e.g., dataset name) and optional overrides
    that map directly to the global configuration variables in config.py.
    Returns:
        argparse.ArgumentParser: The configured parser object.
    """
    parser = argparse.ArgumentParser(
        description="Incremental Structure from Motion (SfM) Pipeline"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset folder inside your DATA_DIR (e.g., proxy_fountain)",
    )
    parser.add_argument(
        "--matcher",
        type=str,
        default="SIFT",
        choices=["SIFT", "SuperPoint", "ORB"],
        help="Feature matching algorithm to use",
    )
    # Optional Configuration Overrides
    parser.add_argument(
        "--downscale_factor",
        type=float,
        default=None,
        help="Override downscale_factor from config.py",
    )
    parser.add_argument(
        "--max_cache_size",
        type=int,
        default=None,
        help="Override max_cache_size from config.py",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=None,
        help="Override max_features from config.py",
    )
    parser.add_argument(
        "--lowe_ratio_threshold",
        type=float,
        default=None,
        help="Override lowe_ratio_threshold from config.py",
    )
    parser.add_argument(
        "--apply_clahe",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override apply_clahe (use --apply_clahe or --no-apply_clahe)",
    )
    parser.add_argument(
        "--disable_symmetric_matching",
        action="store_true",
        help="Disable symmetric cross-check matching",
    )
    parser.add_argument(
        "--use_root_sift",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override use_root_sift (use --use_root_sift or --no-use_root_sift). Only applies when --matcher=SIFT.",
    )
    parser.add_argument(
        "--ransac_prob",
        type=float,
        default=None,
        help="Override ransac_prob from config.py",
    )
    parser.add_argument(
        "--ransac_threshold",
        type=float,
        default=None,
        help="Override ransac_threshold from config.py",
    )
    parser.add_argument(
        "--min_parallax_degrees",
        type=float,
        default=None,
        help="Override min_parallax_degrees from config.py",
    )
    return parser


def main():
    """
    Orchestrates the complete Incremental Structure from Motion (SfM) pipeline execution.
    Workflow:
        1. Parses CLI arguments and applies any optional configuration overrides.
        2. Initializes the IncrementalSfMEngine with the specified dataset and matcher.
        3. Executes the multi-view tracking and mapping loop.
        4. Exports the resulting 3D point cloud as a standard PLY file.
        5. Saves the pipeline configuration and performance metrics to a JSON file.
    """
    parser = build_parser()
    args = parser.parse_args()

    # Apply Optional Overrides to the Global Config Object
    if args.downscale_factor is not None:
        config.downscale_factor = args.downscale_factor
    if args.max_cache_size is not None:
        config.max_cache_size = args.max_cache_size
    if args.max_features is not None:
        config.max_features = args.max_features
    if args.lowe_ratio_threshold is not None:
        config.lowe_ratio_threshold = args.lowe_ratio_threshold
    if args.apply_clahe is not None:
        config.apply_clahe = args.apply_clahe
    if args.disable_symmetric_matching:
        config.disable_symmetric_matching = True
    if args.use_root_sift is not None:
        if args.matcher != "SIFT":
            logger.warning(
                f"The RootSIFT override was passed, but the current matcher is {args.matcher}. "
                "This setting only applies to SIFT and will be ignored."
            )
        config.use_root_sift = args.use_root_sift
    if args.ransac_prob is not None:
        config.ransac_prob = args.ransac_prob
    if args.ransac_threshold is not None:
        config.ransac_threshold = args.ransac_threshold
    if args.min_parallax_degrees is not None:
        config.min_parallax_degrees = args.min_parallax_degrees

    logger.info("=" * 30)
    logger.info("Starting Incremental SfM Pipeline")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Matcher: {args.matcher}")
    logger.info(f"Data Directory: {config.data_dir}")
    logger.info("=" * 30)

    # Initialize and run the core engine
    engine = IncrementalSfMEngine(dataset_name=args.dataset, matcher_type=args.matcher)
    engine.run()

    # Create an 'output' folder directly in the dataset folder
    output_dir = config.data_dir / args.dataset / "output"
    pc_dir = output_dir / "point_clouds"
    metrics_dir = output_dir / "metrics"
    pc_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pc_file = pc_dir / f"point_cloud_{args.matcher}.ply"

    # Export Artifacts
    export_point_cloud_ply(
        filename=str(pc_file), points=engine.point_cloud, colors=engine.point_colors
    )

    metrics_file = metrics_dir / f"run_metrics_{args.matcher}.json"
    run_data = {
        "dataset": args.dataset,
        "matcher": args.matcher,
        "config": {
            "downscale_factor": config.downscale_factor,
            "max_cache_size": config.max_cache_size,
            "max_features": config.max_features,
            "lowe_ratio_threshold": config.lowe_ratio_threshold,
            "apply_clahe": getattr(config, "apply_clahe", False),
            "disable_symmetric_matching": getattr(
                config, "disable_symmetric_matching", False
            ),
            "use_root_sift": getattr(config, "use_root_sift", False),
            "ransac_prob": config.ransac_prob,
            "ransac_threshold": config.ransac_threshold,
            "min_parallax_degrees": config.min_parallax_degrees,
        },
        "results": engine.metrics,
    }

    with open(metrics_file, "w") as f:
        json.dump(run_data, f, indent=4)

    logger.info("=" * 30)
    logger.info("Pipeline Execution Complete")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Matcher: {args.matcher}")
    logger.info(f"Metrics saved to: {metrics_file}")
    logger.info("=" * 30)


if __name__ == "__main__":
    main()
