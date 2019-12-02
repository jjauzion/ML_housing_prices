from src import explore
from src import utils

if __name__ == "__main__":
    args = utils.parse_main_args("file", "conf_output", "force", "verbosity", "edit")
    output_conf = args.conf_output if args.conf_output is not None else args.file
    try:
        dataset = explore.explore_dataset(conf_file=args.file, edit=args.edit, save_file=output_conf)
    except IOError as err:
        print(f'Error: {err}')
        exit(0)
