import h5py

if __name__ == '__main__':
    filepath = '//output/example_wf_time/wind.h5'
    with h5py.File(filepath, 'r') as f:
        for a in f.attrs.items():
            print(f'{a[0]} : {a[1]}')

        print(f['data'][0, :, :, 1])
        print(f['grid'][:, :])
