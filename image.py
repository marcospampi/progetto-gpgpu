import argparse
import cv2
def __mainGetImage():
    arpa = argparse.ArgumentParser()
    arpa.add_argument('--image','-i', required= True, help="Image path", type=str)
    args = arpa.parse_args()

    if 'image' in vars(args):
        cv2.imread( args.image )


if __name__ == '__main__':
    pass