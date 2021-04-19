
def build_external_tracker(args):
    # external tracker wrapper
    if args.external_tracker == "atom" or args.external_tracker == "dimp" or args.external_tracker == "prdimp":
        from .wrapper.atom_dimp.wrapper import TrackerWrapper
        return TrackerWrapper(args)
    else:
        raise ValueError("do not support external tracker called: {}".format(args.external_tracker))

