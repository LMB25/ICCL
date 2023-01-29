from ocpa.algo.discovery.ocpn import algorithm as ocpn_discovery_factory
from ocpa.visualization.oc_petri_net import factory as ocpn_vis_factory

# discover process model for OCEL
def process_discovery_ocel_to_ocpn(ocel_log):
    ocpn = ocpn_discovery_factory.apply(ocel_log, parameters={"debug": False})
    return ocpn

# transform process model to gviz object
def ocpn_to_gviz(ocpn):
    gviz_ocpn = ocpn_vis_factory.apply(ocpn)
    gviz_str = str(gviz_ocpn)
    return gviz_str

# save process model as svg/ png
def save_ocpn(ocpn, filepath, filename, extension):
    ocpn_vis_factory.save(ocpn_vis_factory.apply(ocpn, parameters={'format': extension}), filepath + "/" + filename + "." + extension)