# Latent Coordinate Law Generalization Benchmark

Domains tested: 6
Mean coordinate accuracy: 0.924
Mean all-feature accuracy: 0.962
Blind transfer domain accuracy: 1.000
Mean compression ratio: 0.656
Mean route/map search cost ratio: 409.6
Discovered laws: 1333
Counterexample rows: 4720

## Domain Results

### IGP24
- top coordinates: threshold_zone, a6, a6_abs, a18, a18_abs
- recovered expected coordinates: a6, threshold_zone
- coordinate accuracy: 0.959
- all-feature accuracy: 0.986
- surviving laws: 389 / 406

### ETP
- top coordinates: projection_indicator, variable_count, idempotence_indicator, depth, symmetry
- recovered expected coordinates: depth, idempotence_indicator, projection_indicator
- coordinate accuracy: 0.733
- all-feature accuracy: 0.867
- surviving laws: 18 / 20

### ARC
- top coordinates: symmetry, hole_count, object_count, connected_components, colors
- recovered expected coordinates: connected_components, hole_count, symmetry
- coordinate accuracy: 0.880
- all-feature accuracy: 0.933
- surviving laws: 89 / 145

### MAZE
- top coordinates: graph_nodes, wall_density, grid_size, skeleton_nodes, corridor_width
- recovered expected coordinates: graph_nodes, skeleton_nodes, wall_density
- coordinate accuracy: 0.973
- all-feature accuracy: 1.000
- surviving laws: 75 / 86

### CA
- top coordinates: transition_count, local_entropy, rule_density, rule, mirror_asymmetry
- recovered expected coordinates: local_entropy, mirror_asymmetry, rule_density, transition_count
- coordinate accuracy: 1.000
- all-feature accuracy: 1.000
- surviving laws: 16 / 16

### PHASE
- top coordinates: effective_control, control_parameter, noise, temperature, coupling
- recovered expected coordinates: control_parameter, coupling, effective_control, temperature
- coordinate accuracy: 1.000
- all-feature accuracy: 0.988
- surviving laws: 482 / 660

## Interpretation

The same discovery procedure recovered predictive coordinates, threshold laws, and compressed representations in unrelated generated domains plus IGP24. This supports transfer of the method, not proof of a universal law.
