#!/usr/bin/env python
# -*- coding: UTF-8 -*-

dict_models = ["dispnet", "dispnet_m1", "dispnetcorr", "dispnetcorr_m1", "dispnetcorr_m2", 
                "iresnet", "gcnet", "psmnet"]
def model_create_by_name(name_model, maxdisparity=192):
    model = None
    if name_model == 'dispnet':
        from dispnet import dispnet
        model = dispnet(maxdisparity)
    elif name_model == 'dispnetcorr':
        from dispnetcorr import dispnetcorr
        model = dispnetcorr(maxdisparity)
    elif name_model == 'iresnet':
        from iresnet import iresnet
        model = iresnet(maxdisparity)
    elif name_model == 'gcnet':
        from gcnet import gcnet
        model = gcnet(maxdisparity)
    elif name_model == 'psmnet':
        from psmnet.stackhourglass import PSMNet as psmnet
        model = psmnet(maxdisparity)
    elif name_model == 'dispnet_m1':
        from dispnet_m1 import dispnet_m1
        model = dispnet_m1(maxdisparity)
    elif name_model == 'dispnetcorr_m1':
        from dispnetcorr_m1 import dispnetcorr_m1
        model = dispnetcorr_m1(maxdisparity)
    elif name_model == 'dispnetcorr_m2':
        from dispnetcorr_m2 import dispnetcorr_m2
        model = dispnetcorr_m2(maxdisparity)

    assert model is not None
    return model



