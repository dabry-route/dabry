import os

from dabry.post_processing import PostProcessing


def get_latest_output_dir(base_dir):
    nlist = [dd for dd in os.listdir(base_dir) if
             os.path.isdir(os.path.join(base_dir, dd)) and not dd.startswith('.')]
    latest_subdir = max(nlist, key=lambda name: os.path.getmtime(os.path.join(base_dir, name)))
    return os.path.join(base_dir, latest_subdir)


if __name__ == '__main__':
    base_dir = f'/home/bastien/Documents/work/mermoz/output/'
    output_dir = get_latest_output_dir(base_dir)
    # output_dir = f'/home/bastien/Documents/work/dabry/output/example_energy_band'
    print(output_dir)
    pp = PostProcessing(output_dir)
    pp.load()

    pp.stats()
