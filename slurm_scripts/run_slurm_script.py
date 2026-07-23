from read_depths import gen_pixel_signals, loop_gen_pixel
import argparse
import json

print('Running arg parser', flush=True)

parser = argparse.ArgumentParser()
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'y', 't', '1'):
        return True
    if v.lower() in ('no', 'false', 'n', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument('pixel_x', type=int)
parser.add_argument('pixel_y', type=int)
# parser.add_argument('texp', type=float)

args = vars(parser.parse_args())

for k in args:
    print(args[k], flush=True)


# For 8x8
gen_pixel_signals(px=args['pixel_x'], py=args['pixel_y'])

# For dynamic scene
# loop_gen_pixel(px=args['pixel_x'])