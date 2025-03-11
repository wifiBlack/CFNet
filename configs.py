import argparse
import os

def parse_arguments():
# Argument Parser creation
    parser = argparse.ArgumentParser(
        description="Parameter for data analysis, data cleaning and model training."
    )
    parser.add_argument(
        "--data-dir",
        default="/home/wufan/Datasets/SYSU-CD",
        type=str,
        help="data path",
        required=True
    )
    
    parser.add_argument(
        "--pred-dir",
        type=str,
        help="prediction data path",
        required=False
    )
    
    parser.add_argument(
        "--content-dir",
        type=str,
        help="content output path",
        required=False
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        help="log path",
        required=False
    )

    parser.add_argument(
        "--gpu",
        nargs="+",
        help="id of gpu to use",
        default = [0, 1, 2, 3],
        type=list,
        required=True
    )
    
    parser.add_argument(
        "--epochs",
        help="number of epochs",
        type=int,
        required=False
    )
    
    parser.add_argument(
        "--batch-size",
        help="batch size",
        type=int,
        required=True
    )
    
    parser.add_argument(
        "--num-workers",
        help="number of workers",
        type=int,
        required=True
    )
    
    parser.add_argument(
        "--lr",
        help="learning rate",
        type=float,
        required=False
    )
    
    parser.add_argument(
        "--checkpoint",
        help="Load the trained model",
        type=str,
        required=False
    )

    parsed_arguments = parser.parse_args()

    if parsed_arguments.log_dir is not None:
        # create log dir if it doesn't exists
        if not os.path.exists(parsed_arguments.log_dir):
            os.mkdir(parsed_arguments.log_dir)

        dir_run = sorted(
            [
                filename
                for filename in os.listdir(parsed_arguments.log_dir)
                if filename.startswith("run_")
            ]
        )

        if len(dir_run) > 0:
            num_run = int(dir_run[-1].split("_")[-1]) + 1
        else:
            num_run = 0
        parsed_arguments.log_dir = os.path.join(
            parsed_arguments.log_dir, "run_%04d" % num_run + "/"
        )

    return parsed_arguments