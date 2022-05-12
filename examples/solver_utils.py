import os

def select_wind():
    from ipywidgets import Dropdown
    from IPython.core.display import display
    option_list = sorted(os.listdir('/home/bastien/Documents/data/wind/windy/'))
    new_ol = []
    for e in option_list:
        if not e.endswith('.mz'):
            pass
        else:
            new_ol.append(e[:-3])
    dropdown = Dropdown(description="Choose one:", options=new_ol)
    dropdown.observe(lambda _: 0., names='value')
    display(dropdown)
    return dropdown