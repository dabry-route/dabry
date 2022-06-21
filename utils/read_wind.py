import h5py

if __name__ == '__main__':
    filepath = '/home/bastien/Documents/work/mermoz/output/wf_honolulu-vancouver/wind.h5'
    with h5py.File(filepath, 'r') as f:
        for a in f.attrs.items():
            print(f'{a[0]} : {a[1]}')

        print(f['data'][0, :, :, 1])
        print(f['grid'][:, :])
