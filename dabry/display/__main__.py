import argparse
import os
import sys

#import pandas as pd
#from dash import Input, State, Output, Dash

from .frontend import FrontendHandler

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
    parser.add_argument('-p', '--postprocessing', help='Run post processing', action='store_const',
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
        fh = FrontendHandler(mode='user')
        fh.setup()
        fh.example_dir = args.path
    else:
        fh = FrontendHandler()
        fh.setup()
        fh.select_example(select_latest=args.latest, select_last=args.last)
    if not args.postprocessing:
        fh.run_frontend(block=not args.postprocessing,
                        movie=args.movie,
                        frames=int(args.frames),
                        fps=int(args.fps),
                        flags=args.flags,
                        movie_format=args.format if args.format is not None else 'apng',
                        mini=args.small,
                        mode_3d=args.z)
    else:
        raise Exception('Not implemented')
        # example_dir = os.path.join(fh.output_dir, fh.example_name())
        # pp = PostProcessing(example_dir)
        # app = Dash("Dabry post processing")
        # app.layout = pp.serve_layout
        #
        #
        # @app.callback(
        #     Output('sink', 'children'),
        #     Input('submit-val', 'n_clicks'),
        #     State('table', "derived_virtual_data"),
        #     State('table', 'derived_virtual_selected_rows'),
        # )
        # def update_output(n_clicks, rows, derived_virtual_selected_rows):
        #     if derived_virtual_selected_rows is None:
        #         return ''
        #     print(derived_virtual_selected_rows)
        #     df = pd.DataFrame(rows)
        #     to_hide = []
        #     for i, name in enumerate(df['Name']):
        #         if i not in derived_virtual_selected_rows:
        #             if name not in to_hide:
        #                 to_hide.append(name)
        #     print(to_hide)
        #     with open(os.path.join(example_dir, '.trajfilter'), 'w') as f:
        #         f.writelines(line + '\n' for line in to_hide)
        #     if n_clicks >= 1:
        #         os.system(f'python3 -m dabryvisu {example_dir}')
        #     return f'The input value was "{derived_virtual_selected_rows}" and the button has been clicked {n_clicks} times'
        #
        #
        # app.run_server(debug=False, use_reloader=True)
