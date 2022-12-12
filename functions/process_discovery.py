from ocpa.algo.discovery.ocpn import algorithm as ocpn_discovery_factory
from ocpa.visualization.oc_petri_net import factory as ocpn_vis_factory


def process_discovery_ocel_to_img(ocel_log, img_name):
    ocpn = ocpn_discovery_factory.apply(ocel_log, parameters={"debug": False})
    ocpn_vis_factory.save(ocpn_vis_factory.apply(ocpn), "imgs/" + img_name +".png")

def process_discovery_ocel_to_ocpn(ocel_log):
    ocpn = ocpn_discovery_factory.apply(ocel_log, parameters={"debug": False})
    return ocpn

def ocpn_to_gviz(ocpn):
    gviz_ocpn = ocpn_vis_factory.apply(ocpn)
    gviz_str = str(gviz_ocpn)
    return gviz_str