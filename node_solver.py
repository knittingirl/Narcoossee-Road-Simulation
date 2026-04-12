import numpy as np

def godunov_node_solver(upstream_sending, downstream_receiving, turning_proportions=None): 
  """ Simple Godunov-style node solver for a single merge/diverge node. upstream_sending: array of sending capacities from upstream links (veh/s) downstream_receiving: array of receiving capacities for downstream links (veh/s) turning_proportions: matrix shape (n_up, n_down) with fractions summing to 1 per upstream link Returns flow matrix f[i,j] from upstream i to downstream j """ 
  n_up = upstream_sending.shape[0] 
  n_down = downstream_receiving.shape[0] 
  if turning_proportions is None: # default: all upstream goes to first downstream 
    tp = np.zeros((n_up, n_down)) 
    tp[:, 0] = 1.0 
  else: 
    tp = turning_proportions

  # initial desired flows
  desired = (upstream_sending[:, None] * tp)
  # allocate respecting downstream receiving
  f = np.zeros_like(desired)
  remaining_R = downstream_receiving.copy()
  # simple proportional allocation
  for j in range(n_down):
      demand_to_j = desired[:, j]
      total_demand = demand_to_j.sum()
      if total_demand <= remaining_R[j] + 1e-9:
          f[:, j] = demand_to_j
          remaining_R[j] -= total_demand
      else:
          # scale demands proportionally
          if total_demand > 0:
              f[:, j] = demand_to_j * (remaining_R[j] / total_demand)
              remaining_R[j] = 0.0
  return f
