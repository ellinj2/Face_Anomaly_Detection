from region_proposal_network import RegionProposalNetwork

def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on the region proposal network.")
    parser.add_argument()

    return parser

def main(args):
    model_path = args.model_path

    RegionProposalNetwork()

    pass

if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    main(args)