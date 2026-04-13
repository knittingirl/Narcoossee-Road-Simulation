from dataclasses import dataclass 
import numpy as np

@dataclass 
class Cell: 
  length_m: float 
  lanes: int
  storage_veh: float

class CorridorGeometry: 
  def init(self, n_cells: int, cell_length_m: float, lanes_per_cell: int): 
    self.n_cells = n_cells 
    self.cell_length_m = cell_length_m 
    self.lanes_per_cell = lanes_per_cell 
    self.cells = [Cell(length_m=cell_length_m, lanes=lanes_per_cell, storage_veh=self._default_storage()) 
                  for _ in range(n_cells)]

  def _default_storage(self):
      # rough storage capacity: jam density ~ 150 veh/km/lane
      jam_density_veh_per_m = 150 / 1000.0
      return jam_density_veh_per_m * self.cell_length_m * self.lanes_per_cell
  
  def total_length(self):
      return self.n_cells * self.cell_length_m
  
  def as_dict(self):
      return {"n_cells": self.n_cells, "cell_length_m": self.cell_length_m, "lanes_per_cell": self.lanes_per_cell}
