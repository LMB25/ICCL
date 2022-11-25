from ocpa.algo.conformance.precision_and_fitness import evaluator as quality_measure_factory

def calculate_fitness(ocel, ocpn):
    _, fitness = quality_measure_factory.apply(ocel, ocpn)

    return fitness

def calculate_precision(ocel, ocpn):
    precision, _ = quality_measure_factory.apply(ocel, ocpn)

    return precision