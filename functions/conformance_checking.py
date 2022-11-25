from ocpa.algo.conformance.precision_and_fitness import evaluator as quality_measure_factory
import pandas as pd

def calculate_fitness(ocel, ocpn):
    _, fitness = quality_measure_factory.apply(ocel, ocpn)

    return fitness

def calculate_precision(ocel, ocpn):
    precision, _ = quality_measure_factory.apply(ocel, ocpn)

    return precision


def create_conformance_df(conformance_result, measure):

    conformance_df = pd.DataFrame(columns=['Cluster', measure])
    num_clusters = len(conformance_result)
    clusters = [i for i in range(1, num_clusters + 1)]
    conformance_df['Cluster'] = clusters
    conformance_df[measure] = conformance_result

    return conformance_df