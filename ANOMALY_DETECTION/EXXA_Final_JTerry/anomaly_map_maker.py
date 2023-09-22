import argparse

from utils.globals import crops
from utils.inference_utils import get_fits_in_dir, make_anomaly_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Inference",
        description="Makes anomaly maps",
    )

    parser.add_argument("--model_name", type=str, default="valiant-sweep-5", help="Trained model")
    parser.add_argument(
        "--test_path", type=str, default="./Data/Non_Keplerian/Planets/", help="Path to test data"
    )
    parser.add_argument(
        "--fits_name",
        type=str,
        default="13CO_lines_run_0_planet0_00226.fits",
        help="Name of fits file to use",
    )
    parser.add_argument(
        "--over_all_fits", type=int, default=0, help="Make map for all fits in directory"
    )
    parser.add_argument(
        "--inference_batches",
        type=int,
        default=0,
        help="Size of batches to do inference on (0 = all at once)",
    )
    parser.add_argument("--line_index", type=int, default=1, help="Which emission line to use")
    parser.add_argument(
        "--max_seq_length", type=int, default=0, help="Maximum spectrum length (0 = auto)"
    )
    parser.add_argument(
        "--model_path", type=str, default="./trained_models/", help="Directory of trained models"
    )
    parser.add_argument(
        "--use_checkpoint",
        type=int,
        default=0,
        help="Use saved checkpoint rather than final model",
    )
    parser.add_argument("--accelerator_name", type=str, default="cpu", help="Accelerator to use")
    parser.add_argument(
        "--max_v", type=float, default=3.0, help="Maximum spectral velocity to consider"
    )
    parser.add_argument("--sub_cont", type=int, default=1, help="Subtract continuum")
    parser.add_argument(
        "--save_dir", type=str, default="./Results/", help="Where to save results"
    )
    parser.add_argument("--save", type=int, default=1, help="Save map?")
    parser.add_argument("--crop_min_x", type=int, default=0, help="Crop minimum x (0 = no crop)")
    parser.add_argument("--crop_max_x", type=int, default=0, help="Crop maximum x (0 = no crop)")
    parser.add_argument("--crop_min_y", type=int, default=0, help="Crop minimum y")
    parser.add_argument("--crop_max_y", type=int, default=0, help="Crop maximum y")
    parser.add_argument(
        "--scale_data",
        type=float,
        default=-1.0,
        help="Amount by which to scale data (-1 = normalize)",
    )
    parser.add_argument("--autocrop", type=int, default=1, help="Use predefined croping")
    parser.add_argument(
        "--parameter_path",
        type=str,
        default="./Parameters/",
        help="Where to save model parameters",
    )

    args = parser.parse_args()

    crop_min_x = args.crop_min_x
    crop_min_y = args.crop_min_y
    crop_max_x = args.crop_max_x
    crop_max_y = args.crop_max_y

    crop = (
        (crop_min_x if crop_min_x != 0 else None, crop_max_x if crop_max_x != 0 else None),
        (crop_min_y if crop_min_y != 0 else None, crop_max_y if crop_max_y != 0 else None),
    )

    if bool(args.autocrop) and args.fits_name in crops:
        crop = crops[args.fits_name]

    return_data = False

    if not bool(args.over_all_fits):
        fits_names = [args.fits_name]
        test_paths = [args.test_path]

    else:
        fits_names, test_paths = get_fits_in_dir(test_path=args.test_path)

    for fits_name, test_path in zip(fits_names, test_paths, strict=True):
        make_anomaly_map(
            test_path=test_path,
            fits_name=fits_name,
            crop=crop,
            max_v=args.max_v,
            max_seq_length=args.max_seq_length,
            save=bool(args.save),
            scale_data=args.scale_data,
            model_name=args.model_name,
            model_path=args.model_path,
            use_checkpoint=bool(args.use_checkpoint),
            accelerator_name=args.accelerator_name,
            save_dir=args.save_dir,
            return_data=return_data,
            sub_cont=bool(args.sub_cont),
            inference_batches=args.inference_batches,
            parameter_path=args.parameter_path,
        )
