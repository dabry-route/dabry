import argparse
import sys

from dabry.display.display import Display

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trajectory planning display tool')
    kwstore_cost = {
        'action': 'store_const',
        'const': True,
        'default': False
    }
    parser.add_argument('path', help='Path to source for display', nargs='?')
    parser.add_argument('-l', '--latest', help='Run most recent results', action='store_const',
                        const=True, default=False)
    parser.add_argument('-L', '--last', help='Run last opened results', action='store_const',
                        const=True, default=False)
    parser.add_argument('-m', '--movie', help='Produce movie with case', action='store_const',
                        const=True, default=False)
    parser.add_argument('-z', help='Use 3d representation (for energy optimal problems)', action='store_const',
                        const=True, default=False)
    parser.add_argument('-f', '--format', help='Format for movie rendering', required=False, default=None)
    parser.add_argument('-s', '--small', help='Render only map frame', default=False,
                        action='store_const', const=True)
    parser.add_argument('--frames', help='Number of frames for movie', default=50)
    parser.add_argument('--fps', help='Framerate for movie', default=10)
    parser.add_argument('--flags', help='Flags for display', default='')
    args = parser.parse_args(sys.argv[1:])

    if args.path is not None:
        display = Display.from_path(args.path, latest=args.latest, last=args.last, mode_3d=args.z)
    else:
        display = Display.from_scratch(latest=args.latest, last=args.last, mode_3d=args.z)
    display.run(movie=args.movie,
                frames=int(args.frames),
                fps=int(args.fps),
                flags=args.flags,
                movie_format=args.format if args.format is not None else 'apng',
                mini=args.small)
