import json
import os
import sys
from math import pi
from pathlib import Path

import geopy
from geopy import Nominatim

"""
geodata.py
Handles translation from natural language to coordinates on Earth.
Handles the retrieval and caching of data.

Copyright (C) 2021 Bastien Schnitzler 
(bastien dot schnitzler at live dot fr)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


class GeoData:

    def __init__(self):
        self.cache_dir = os.path.join(str(Path.home()), '.cache', 'geodata')
        self.cache_path = os.path.join(self.cache_dir, 'geodata.json')
        self.full_inet = True
        if not os.path.isdir(self.cache_dir):
            try:
                os.mkdir(self.cache_dir)
                self.full_inet = False
            except:
                pass
            self.cache = {}
        else:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'r') as f:
                    self.cache = json.load(f)
                    self.full_inet = False
            else:
                self.cache = {}
        if self.full_inet:
            print('GeoData : unable to locate cache, switching to full-Internet mode')
        self.locator = None

    def get_coords(self, name, units='deg'):
        """
        Get coordinates for corresponding geocode
        :param name: The geodcode
        :return: Coordinates (lon, lat) in degrees
        """
        if units not in ['deg', 'rad']:
            print(f'Unknown units "{units}"', file=sys.stderr)
            exit(1)
        name = name.lower().strip()
        if not self.full_inet and name in self.cache.keys():
            res_deg = self.cache[name]
        else:
            def get_res(force_proxy=False):
                kwargs = {'user_agent': 'openstreetmaps'}
                if force_proxy:
                    kwargs['proxies'] = 'http://proxy:3128/'
                self.locator = Nominatim(**kwargs)
                loc = self.locator.geocode(name)
                point = (loc.longitude, loc.latitude)
                if not self.full_inet:
                    self.cache[name] = point
                    with open(self.cache_path, 'w') as f:
                        json.dump(self.cache, f)
                return point

            try:
                res_deg = get_res()
            except geopy.exc.GeocoderUnavailable:
                res_deg = get_res(force_proxy=True)

        if units == 'rad':
            return [pi / 180. * res_deg[0], pi / 180. * res_deg[1]]
        else:
            return res_deg


if __name__ == '__main__':
    gd = GeoData()
    print(gd.get_coords('Dakar'))
    print(gd.get_coords('Natal'))
