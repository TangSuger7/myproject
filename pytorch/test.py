# -*- coding: UTF-8 -*-
from IPython.display import display,Image
image_path = '链表.png'
display(Image(filename=image_path,embed=True))

class Json(object):
    def __init__(self, json):
        self.json = json
    def _repr_pretty_(self, pp, cycle):
        import json
        pp.text(json.dumps(self.json, indent=2))
    def __repr__(self):
        return str(self.json)
d = Json({1:2, 3: {4:5}})
display(d)