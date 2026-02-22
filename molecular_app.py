import sys
import os
import sqlite3
import math
import traceback
from datetime import datetime, timezone
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import time

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint, QTimer, QObject, QMutex
from PyQt6.QtGui import QVector3D, QFont, QPainter, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QLineEdit, QTabWidget, QTableWidget, QTableWidgetItem,
    QTextEdit, QMessageBox, QSplitter, QDoubleSpinBox, QProgressBar, QComboBox,
    QSlider, QSpinBox, QCheckBox, QFrame, QGridLayout, QFrame
)

import pyqtgraph as pg
import pyqtgraph.opengl as gl

# --- Scientific Constants ---
AMU_TO_KG = 1.66053906660e-27  # Conversion factor from atomic mass unit to kilogram
ANG_TO_M = 1e-10               # Conversion factor from angstrom to meter
ANG_PER_PS_TO_M_PER_S = 1e-10 / 1e-12 # Conversion for velocity from Å/ps to m/s
BOLTZMANN_K = 1.380649e-23     # Boltzmann constant in Joules per Kelvin
COULOMB_K = 8.9875517923e9     # Coulomb's constant in N·m²/C²
E_CHARGE = 1.602176634e-19     # Elementary charge in Coulombs
EPSILON_J = 1.660e-21          # Lennard-Jones potential well depth for C-C, in Joules
SIGMA_M = 3.4e-10              # Lennard-Jones distance at zero energy for C-C, in meters
MAX_FORCE_LIMIT = 5e-11        # A constant to prevent force overflow in calculations
BOUNDARY_LIMIT = 5e-11         # A constant to limit forces at the boundary
BOUNDARY_DISTANCE = 10.0       # Distance from the center where boundary forces kick in (in Angstroms)

# --- CPK Coloring Scheme ---
CPK_COLORS = {
    'H': (1.0, 1.0, 1.0, 1.0),     # White for Hydrogen
    'C': (0.2, 0.2, 0.2, 1.0),     # Dark gray for Carbon
    'N': (0.0, 0.0, 1.0, 1.0),     # Blue for Nitrogen
    'O': (1.0, 0.0, 0.0, 1.0),     # Red for Oxygen
    'F': (0.0, 1.0, 0.0, 1.0),     # Green for Fluorine
    'CL': (0.0, 1.0, 0.0, 1.0),    # Green for Chlorine
    'BR': (0.6, 0.13, 0.0, 1.0),   # Brown for Bromine
    'I': (0.58, 0.0, 0.58, 1.0),   # Purple for Iodine
    'P': (1.0, 0.5, 0.0, 1.0),     # Orange for Phosphorus
    'S': (1.0, 1.0, 0.18, 1.0),    # Yellow for Sulfur
    'FE': (0.87, 0.47, 0.0, 1.0),  # Orange for Iron
    'NA': (0.6, 0.0, 0.6, 1.0),    # Purple for Sodium
    'K': (0.6, 0.0, 0.6, 1.0),     # Purple for Potassium
    'CA': (0.6, 0.0, 0.6, 1.0),    # Purple for Calcium
}

# --- Atomic Radii and Masses ---
VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
    'P': 1.80, 'S': 1.80, 'CL': 1.75, 'BR': 1.85, 'I': 1.98, 'FE': 1.94,
    'NA': 2.27, 'K': 2.75, 'CA': 2.23
}

COV_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'P': 1.07, 'S': 1.05,
    'CL': 1.02, 'BR': 1.20, 'I': 1.39, 'NA': 1.66, 'K': 2.03, 'CA': 1.76, 'FE': 1.26
}

ATOMIC_MASSES = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'S': 32.06, 'P': 30.974, 'FE': 55.845,
    'NA': 22.990, 'K': 39.098, 'CA': 40.078
}

# --- Data Loading and Processing Functions ---
def load_csv(path: str) -> pd.DataFrame:
    """Loads a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        required = ['Atom_ID', 'Element', 'X', 'Y', 'Z']
        for r in required:
            if r not in df.columns:
                raise ValueError(f"CSV missing required column: {r}")
        df['Element'] = df['Element'].astype(str).str.strip().str.upper()
        for col in ['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz', 'Charge', 'Mass(amu)']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at: {path}")
    except Exception as e:
        raise ValueError(f"Error loading CSV: {e}")

def fill_masses(df: pd.DataFrame, mass_col: str = 'Mass(amu)') -> pd.DataFrame:
    """Fills in missing atomic mass values based on the element symbol."""
    if mass_col not in df.columns:
        df[mass_col] = np.nan
    for i, row in df[df[mass_col].isna()].iterrows():
        el = str(row['Element']).upper()
        m = ATOMIC_MASSES.get(el)
        if m is None:
            m = ATOMIC_MASSES.get(el[:2]) or ATOMIC_MASSES.get(el[0]) or 12.011
        df.at[i, mass_col] = m
    df[mass_col] = df[mass_col].fillna(12.011)
    return df

def coords_array(df: pd.DataFrame) -> np.ndarray:
    """Extracts X, Y, Z coordinates into a NumPy array."""
    return df[['X', 'Y', 'Z']].to_numpy(dtype=float)

def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Computes a matrix of all pairwise distances between atoms."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    return dists

def detect_bonds(df: pd.DataFrame, tolerance: float = 0.45) -> List[Tuple[str, str, float]]:
    """Detects bonds based on a simple covalent radius-based cutoff criterion."""
    coords = coords_array(df)
    dmat = compute_distance_matrix(coords)
    n = len(df)
    bonds = []
    elems = df['Element'].tolist()
    for i in range(n):
        for j in range(i + 1, n):
            el_i, el_j = elems[i], elems[j]
            r_i = COV_RADII.get(el_i, COV_RADII.get(el_i[:2], 0.75))
            r_j = COV_RADII.get(el_j, COV_RADII.get(el_j[:2], 0.75))
            cutoff = r_i + r_j + tolerance
            dij = dmat[i, j]
            if dij <= cutoff and dij > 0:
                bonds.append((str(df.at[i, 'Atom_ID']), str(df.at[j, 'Atom_ID']), float(dij)))
    return bonds

def compute_kinetic_energy(df: pd.DataFrame, mass_col: str = 'Mass(amu)') -> pd.Series:
    """Calculates the kinetic energy for each atom in Joules."""
    vx = df.get('Vx', pd.Series(0, index=df.index)).fillna(0).to_numpy(dtype=float)
    vy = df.get('Vy', pd.Series(0, index=df.index)).fillna(0).to_numpy(dtype=float)
    vz = df.get('Vz', pd.Series(0, index=df.index)).fillna(0).to_numpy(dtype=float)
    v_ang = np.sqrt(vx**2 + vy**2 + vz**2)
    v_mps = v_ang * ANG_PER_PS_TO_M_PER_S
    m_amu = df[mass_col].to_numpy(dtype=float)
    m_kg = m_amu * AMU_TO_KG
    ke = 0.5 * m_kg * (v_mps ** 2)
    return pd.Series(ke, index=df.index)

def compute_coulomb_pe(df: pd.DataFrame) -> pd.Series:
    """Calculates the Coulomb potential energy for each atom in Joules."""
    n = len(df)
    charges = df.get('Charge', pd.Series(0, index=df.index)).fillna(0).to_numpy(dtype=float) * E_CHARGE
    coords = coords_array(df)
    dmat_m = compute_distance_matrix(coords) * ANG_TO_M
    pe_per_atom = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            r = dmat_m[i, j]
            if r <= 0: continue
            pe_per_atom[i] += COULOMB_K * charges[i] * charges[j] / r
    return pd.Series(pe_per_atom, index=df.index)

def kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Aligns two sets of coordinates using the Kabsch algorithm."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(Wt.T @ V.T))
    D = np.diag([1.0, 1.0, d])
    U = Wt.T @ D @ V.T
    P_rot = Pc @ U
    return P_rot + Q.mean(axis=0)

def compute_rmsd(df_ref: pd.DataFrame, df_target: pd.DataFrame) -> float:
    """Calculates the Root Mean Square Deviation (RMSD) between two structures."""
    P = coords_array(df_ref)
    Q = coords_array(df_target)
    if P.shape != Q.shape:
        raise ValueError("Reference and target must have same number/order of atoms for RMSD")
    P_aligned = kabsch_align(P, Q)
    rmsd = math.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1)))
    return float(rmsd)

# --- Database Interaction Functions ---
def init_db(conn: sqlite3.Connection):
    """Initializes the SQLite database tables."""
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS simulations (
            sim_id INTEGER PRIMARY KEY, name TEXT, timestamp TEXT, source_file TEXT, notes TEXT
        )""")
    c.execute("""
        CREATE TABLE IF NOT EXISTS atoms (
            id INTEGER PRIMARY KEY, sim_id INTEGER, atom_id TEXT, element TEXT,
            x REAL,y REAL,z REAL, vx REAL, vy REAL, vz REAL, charge REAL, mass_amu REAL,
            ke_joule REAL, pe_joule REAL
        )""")
    c.execute("""
        CREATE TABLE IF NOT EXISTS bonds (
            id INTEGER PRIMARY KEY, sim_id INTEGER, atom1 TEXT, atom2 TEXT, length_ang REAL
        )""")
    c.execute("""
        CREATE TABLE IF NOT EXISTS summary (
            sim_id INTEGER PRIMARY KEY, n_atoms INTEGER, n_bonds INTEGER,
            total_ke REAL, total_pe REAL, avg_bond_length REAL, rmsd REAL
        )""")
    conn.commit()

def save_simulation(conn: sqlite3.Connection, name: str, source_file: str, notes: str='') -> int:
    """Inserts a new simulation record into the database."""
    ts = datetime.now(timezone.utc).isoformat()
    c = conn.cursor()
    c.execute("INSERT INTO simulations (name,timestamp,source_file,notes) VALUES (?,?,?,?)",
              (name, ts, source_file, notes))
    conn.commit()
    return c.lastrowid

def save_atoms(conn: sqlite3.Connection, sim_id: int, df: pd.DataFrame, ke: pd.Series, pe: pd.Series, mass_col: str = 'Mass(amu)'):
    """Saves atom data for a given simulation ID."""
    c = conn.cursor()
    for idx, row in df.iterrows():
        c.execute("""
            INSERT INTO atoms (sim_id, atom_id, element, x, y, z, vx, vy, vz, charge, mass_amu, ke_joule, pe_joule)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            sim_id, str(row['Atom_ID']), str(row['Element']),
            float(row['X']), float(row['Y']), float(row['Z']),
            float(row.get('Vx', 0.0)),
            float(row.get('Vy', 0.0)),
            float(row.get('Vz', 0.0)),
            float(row.get('Charge', 0.0)),
            float(row.get(mass_col, 0.0)),
            float(ke.at[row.name]), float(pe.at[row.name])
        ))
    conn.commit()

def save_bonds(conn: sqlite3.Connection, sim_id: int, bonds: List[Tuple[str, str, float]]):
    """Saves detected bond data for a given simulation ID."""
    c = conn.cursor()
    for a1, a2, length in bonds:
        c.execute("INSERT INTO bonds (sim_id, atom1, atom2, length_ang) VALUES (?,?,?,?)",
                  (a1, a2, length))
    conn.commit()

def save_summary(conn: sqlite3.Connection, sim_id: int, n_atoms: int, n_bonds: int, total_ke: float, total_pe: float, avg_bond: float, rmsd_val: float):
    """Saves a summary of the simulation results."""
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO summary (sim_id, n_atoms, n_bonds, total_ke, total_pe, avg_bond_length, rmsd)
        VALUES (?,?,?,?,?,?,?)
    """, (sim_id, n_atoms, n_bonds, total_ke, total_pe, avg_bond, rmsd_val if rmsd_val is not None else 0.0))
    conn.commit()

# --- NEW: Shared Data Manager for Thread-Safe Communication ---
class SharedDataManager:
    """A thread-safe container for passing data from the worker to the GUI."""
    def __init__(self):
        self._lock = QMutex()
        self._data = {'step': 0, 'df': None, 'ke': 0.0, 'pe': 0.0, 'bonds': []}
        
    def update_data(self, step, df, ke, pe, bonds):
        """Updates the internal data with new simulation results."""
        self._lock.lock()
        self._data['step'] = step
        self._data['df'] = df.copy()
        self._data['ke'] = ke.sum()
        self._data['pe'] = pe.sum() / 2.0
        self._data['bonds'] = bonds
        self._lock.unlock()

    def get_data(self):
        """Retrieves the latest simulation data."""
        self._lock.lock()
        data = self._data.copy()
        self._lock.unlock()
        return data

# --- NEW: Advanced Force Field Classes ---
class ForceField:
    """Calculates forces and potential energy for a molecular system."""
    def __init__(self, df: pd.DataFrame, bonds: List[Tuple[str, str, float]]):
        self.masses = df['Mass(amu)'].to_numpy(dtype=float) * AMU_TO_KG
        self.charges = df.get('Charge', pd.Series(0, index=df.index)).fillna(0).to_numpy(dtype=float) * E_CHARGE
        self.atom_types = df['Element'].tolist()
        
        self.bonds_indices = self._get_bond_indices(df, bonds)
        self.angles_indices = self._get_angle_indices(df, self.bonds_indices)

        # Force field parameters for C-C, C-H, and other common bonds/angles.
        self.morse_D_e = 4.74  # C-C bond energy (eV)
        self.morse_alpha = 1.6  # C-C parameter
        self.morse_r_e = 1.54 # C-C equilibrium distance (Ang)
        self.angle_k = 0.005 # Angle bending force constant (J/rad^2)
        self.angle_theta0 = 109.5 * np.pi / 180.0 # Ideal angle (deg to rad)
        self.vdw_epsilon = 1.660e-21 # VDW energy parameter
        self.vdw_sigma = 3.4e-10 # VDW distance parameter
        self.pe_per_atom = np.zeros(len(df), dtype=float)

    def _get_bond_indices(self, df: pd.DataFrame, bonds: List[Tuple[str, str, float]]):
        """Maps Atom_IDs to their integer indices for faster lookup."""
        bond_map = {str(aid): i for i, aid in enumerate(df['Atom_ID'])}
        indices = []
        for a1, a2, _ in bonds:
            if a1 in bond_map and a2 in bond_map:
                indices.append((bond_map[a1], bond_map[a2]))
        return indices
    
    def _get_angle_indices(self, df: pd.DataFrame, bonds: List[Tuple[int, int]]):
        """Identifies atom triplets that form a bond angle."""
        adjacency = {i: [] for i in range(len(df))}
        for i, j in bonds:
            adjacency[i].append(j)
            adjacency[j].append(i)
        
        angles = []
        for i in range(len(df)):
            for j in adjacency[i]:
                for k in adjacency[i]:
                    if j < k:
                        angles.append((j, i, k))
        return angles

    def calculate(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates forces and potential energy from all force field components."""
        n_atoms = len(coords)
        forces = np.zeros_like(coords, dtype=float)
        self.pe_per_atom.fill(0.0) # Reset potential energy at each step

        coords_m = coords * ANG_TO_M

        # Non-bonded forces (Lennard-Jones & Coulomb)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_vec = coords_m[j] - coords_m[i]
                r = np.linalg.norm(r_vec)
                r = max(r, 1e-15) # Prevent division by zero

                # Lennard-Jones Attractive Term
                r_6 = (self.vdw_sigma / r)**6
                lj_pe = -4 * self.vdw_epsilon * r_6  # Only attractive term
                lj_force_mag = 24 * self.vdw_epsilon / r * r_6  # Only attractive term
                
                # total_force_mag = coulomb_force_mag + lj_force_mag # Removed Coulomb force

                total_force_mag = lj_force_mag # Only Lennard-Jones attractive term
                
                # Clamp the total force magnitude to prevent numerical instability
                total_force_mag = np.clip(total_force_mag, -MAX_FORCE_LIMIT, MAX_FORCE_LIMIT)

                total_force_vec = total_force_mag * (r_vec / r)
                forces[i] -= total_force_vec
                forces[j] += total_force_vec
                # pe_per_atom[i] += 0.5 * (coulomb_pe + lj_pe) # Removed Coulomb PE
                self.pe_per_atom[i] += 0.5 * (lj_pe)
                self.pe_per_atom[j] += 0.5 * (lj_pe)

        # Bonded forces (Morse Potential)
        for i, j in self.bonds_indices:
            r_vec = coords_m[j] - coords_m[i]
            r = np.linalg.norm(r_vec)
            r = max(r, 1e-15) # Prevent division by zero
            r_ang = r / ANG_TO_M
            
            # Morse potential energy
            morse_pe = self.morse_D_e * (1 - np.exp(-self.morse_alpha * (r_ang - self.morse_r_e)))**2 * E_CHARGE
            
            # Morse force
            morse_force_mag = -2 * self.morse_D_e * self.morse_alpha * (1 - np.exp(-self.morse_alpha * (r_ang - self.morse_r_e))) * np.exp(-self.morse_alpha * (r_ang - self.morse_r_e)) * E_CHARGE
            
            # Clamp the morse force magnitude as well
            morse_force_mag = np.clip(morse_force_mag, -MAX_FORCE_LIMIT, MAX_FORCE_LIMIT)

            morse_force_vec = morse_force_mag * (r_vec / r)
            
            forces[i] -= morse_force_vec
            forces[j] += morse_force_vec
            self.pe_per_atom[i] += 0.5 * morse_pe
            self.pe_per_atom[j] += 0.5 * morse_pe

        # Angle bending forces
        for i, j, k in self.angles_indices:
            r_ij = coords_m[j] - coords_m[i]
            r_ik = coords_m[k] - coords_m[i]
            r_ij_norm = np.linalg.norm(r_ij)
            r_ik_norm = np.linalg.norm(r_ik)
            
            if r_ij_norm * r_ik_norm > 1e-15:
                cos_theta = np.dot(r_ij, r_ik) / (r_ij_norm * r_ik_norm)
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                
                # Angle bending potential energy
                angle_pe = 0.5 * self.angle_k * (theta - self.angle_theta0)**2
                self.pe_per_atom[i] += angle_pe / 2.0
                self.pe_per_atom[j] += angle_pe / 2.0
                self.pe_per_atom[k] += angle_pe / 2.0

                # Derivatives for force calculation
                d_theta_d_rij = (r_ij_norm**-1) * (r_ik - r_ij * cos_theta) / np.sin(theta)
                d_theta_d_rik = (r_ik_norm**-1) * (r_ij - r_ik * cos_theta) / np.sin(theta)

                force_mag = -self.angle_k * (theta - self.angle_theta0)
                
                # Clamp angle force magnitude
                force_mag = np.clip(force_mag, -MAX_FORCE_LIMIT, MAX_FORCE_LIMIT)

                force_i = force_mag * (d_theta_d_rij + d_theta_d_rik)
                
                forces[i] -= force_i

        # Apply boundary force to keep atoms within view
        for i, pos in enumerate(coords):
            dist_from_center = np.linalg.norm(pos)
            if dist_from_center > BOUNDARY_DISTANCE:
                # Directional vector from the center to the atom
                direction_vec = pos / dist_from_center
                
                # Calculate a repulsive force that increases with distance
                boundary_force_mag = BOUNDARY_LIMIT * (dist_from_center - BOUNDARY_DISTANCE)
                boundary_force_vec = boundary_force_mag * direction_vec
                
                forces[i] -= boundary_force_vec * ANG_TO_M

        return forces, self.pe_per_atom

# --- Thermostat Class (Berendsen) ---
class BerendsenThermostat:
    """Applies a Berendsen thermostat to maintain a target temperature."""
    def __init__(self, target_temp, tau=0.1):
        self.target_temp = target_temp
        self.tau = tau
    
    def apply(self, velocities, masses, dt):
        n_atoms = len(masses)
        dof = 3 * n_atoms - 3 # Degrees of freedom for a non-constrained system
        
        ke = 0.5 * np.sum(masses[:, None] * velocities**2)
        current_temp = (2 * ke) / (dof * BOLTZMANN_K)
        
        if current_temp == 0:
            current_temp = 1e-6 # Avoid division by zero
        
        scaling_factor = np.sqrt(1 + (dt / self.tau) * ((self.target_temp / current_temp) - 1))
        
        velocities *= scaling_factor
        return velocities

# --- Simulation Engine Logic ---
class SimulationEngine(QObject):
    """The core simulation logic, now a QObject for signaling."""
    # This signal is now only used for the end of the simulation, not for updates
    finished = pyqtSignal()

    def __init__(self, start_df: pd.DataFrame, bonds: List[Tuple[str,str,float]], tolerance: float, data_manager: SharedDataManager, temperature: float = 300.0):
        super().__init__()
        self.df = start_df.copy()
        self.bonds = bonds
        self.tolerance = tolerance
        self.data_manager = data_manager # NEW: Instance of shared data manager
        self.temperature = temperature
        self.DT_S = 0.001 * 1e-12  # Hardcoded timestep in seconds
        
        if 'Vx' not in self.df.columns: self.df['Vx'] = 0.0
        if 'Vy' not in self.df.columns: self.df['Vy'] = 0.0
        if 'Vz' not in self.df.columns: self.df['Vz'] = 0.0

        self.positions = self.df[['X', 'Y', 'Z']].to_numpy(dtype=float) * ANG_TO_M
        self.velocities = self.df[['Vx', 'Vy', 'Vz']].to_numpy(dtype=float) * ANG_PER_PS_TO_M_PER_S
        self.masses = self.df['Mass(amu)'].to_numpy(dtype=float) * AMU_TO_KG

        self.force_calculator = ForceField(self.df, self.bonds)
        self.forces, self.pe_per_atom = self.force_calculator.calculate(self.positions / ANG_TO_M)
        self.thermostat = BerendsenThermostat(self.temperature)
        
        self.is_running = False

    def run(self):
        """Starts the infinite simulation loop."""
        self.is_running = True
        step_count = 0
        while self.is_running:
            try:
                # 1. Update positions (first half of Velocity Verlet)
                self.positions += self.velocities * self.DT_S + 0.5 * (self.forces / self.masses[:, None]) * self.DT_S**2

                # 2. Re-calculate forces
                forces_new, self.pe_per_atom = self.force_calculator.calculate(self.positions / ANG_TO_M)

                # 3. Update velocities (second half of Velocity Verlet)
                self.velocities += 0.5 * ((self.forces + forces_new) / self.masses[:, None]) * self.DT_S

                # 4. Apply thermostat
                self.velocities = self.thermostat.apply(self.velocities, self.masses, self.DT_S)

                # Update forces for next iteration
                self.forces = forces_new

                # Update DataFrame with new positions and velocities
                self.df[['X', 'Y', 'Z']] = self.positions / ANG_TO_M
                self.df[['Vx', 'Vy', 'Vz']] = self.velocities / ANG_PER_PS_TO_M_PER_S
                
                step_count += 1
                
                # NEW: Periodically update the shared data manager
                if step_count % 100 == 0:
                    ke_per_atom = compute_kinetic_energy(self.df)
                    pe_series = pd.Series(self.pe_per_atom, index=self.df.index)
                    bonds = detect_bonds(self.df, tolerance=self.tolerance)
                    self.data_manager.update_data(step_count, self.df, ke_per_atom, pe_series, bonds)
                
            except Exception as e:
                print(f"Simulation error: {e}")
                traceback.print_exc()
                self.is_running = False
                break
        
        self.finished.emit()
            
    def stop(self):
        self.is_running = False

# --- OpenGL Visualization Helper Functions ---
def bake_phong_vertex_colors(meshdata: gl.MeshData, base_color: tuple,
                             light_dir=(0.6, 0.6, 0.4),
                             ambient=0.18, diffuse_k=0.75, specular_k=0.3, shininess=24):
    """Bakes vertex colors with Phong shading for a realistic look."""
    verts = np.array(meshdata.vertexes(), dtype=float)
    try:
        normals = np.array(meshdata.vertexNormals(), dtype=float)
    except Exception:
        normals = verts.copy()
        norms = np.linalg.norm(normals, axis=1)
        norms[norms == 0] = 1.0
        normals = normals / norms[:, None]

    ld = np.array(light_dir, dtype=float)
    ld = ld / np.linalg.norm(ld)
    base_rgb = np.array(base_color[:3], dtype=float)
    colors = np.zeros((len(verts), 4), dtype=np.float32)
    view_dir = np.array([0.0, 0.0, 1.0])
    for i, n in enumerate(normals):
        n = n / (np.linalg.norm(n) + 1e-12)
        diff = max(np.dot(n, ld), 0.0)
        r = 2.0 * np.dot(n, ld) * n - ld
        spec = max(np.dot(r, view_dir), 0.0) ** shininess
        rgb = ambient * base_rgb + diffuse_k * diff * base_rgb + specular_k * spec * np.ones(3)
        rgb = np.clip(rgb, 0.0, 1.0)
        colors[i, :3] = rgb
        colors[i, 3] = base_color[3] if len(base_color) > 3 else 1.0
    return colors

def make_shaded_sphere_mesh(element: str, rows: int = 24, cols: int = 24):
    """Creates a sphere mesh with pre-calculated vertex colors for shading."""
    md = gl.MeshData.sphere(rows=rows, cols=cols, radius=1.0)
    base_color = CPK_COLORS.get(element, (0.6, 0.6, 0.6, 1.0))
    try:
        vc = bake_phong_vertex_colors(md, base_color)
        md.setVertexColors(vc)
    except Exception:
        pass
    return md, base_color

# --- Main Application Window and GUI ---
class MainWindow(QMainWindow):
    """The main window of the molecular simulation application."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FINAL BOSS — Molecular Simulator")
        self.resize(1280, 860)

        self.csv_path = None
        self.ref_path = None
        self.db_path = os.path.abspath("molecular_data.db")
        self.sim_name = "final_boss_run"
        self.tolerance = 0.45

        self.last_df = None
        self.last_bonds = None
        self.last_ke = None
        self.last_pe = None
        self.last_sim_id = None
        
        self.selected_atom_idx = -1
        
        self.sim_thread = None
        self.engine = None
        self.data_manager = SharedDataManager() # NEW: Instance of shared data manager
        
        self.time_step_count = 0
        
        self.energy_plot_data = {'time': [], 'ke': [], 'pe': [], 'total': []}

        # NEW: Timer for GUI updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.on_update)
        
        self.atoms_are_rendered = False
        self.sphere_items = []
        self.line_items = []
        self.grid_item = None
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        central.setLayout(layout)

        # Main Controls Layout
        top = QHBoxLayout()
        self.btn_load = QPushButton("Load CSV")
        self.lbl_csv = QLabel("No CSV loaded")
        self.btn_ref = QPushButton("Load Ref (opt)")
        self.lbl_ref = QLabel("No ref")
        self.db_input = QLineEdit(self.db_path)
        self.db_input.setFixedWidth(240)
        self.name_input = QLineEdit(self.sim_name)
        self.tol_input = QDoubleSpinBox()
        self.tol_input.setRange(0.0, 5.0)
        self.tol_input.setSingleStep(0.05)
        self.tol_input.setValue(self.tolerance)
        
        self.btn_run = QPushButton("Run Simulation")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.btn_export = QPushButton("Export Summary TXT")
        self.btn_open_db = QPushButton("Open DB (external viewer)")

        top.addWidget(self.btn_load)
        top.addWidget(self.lbl_csv)
        top.addWidget(self.btn_ref)
        top.addWidget(self.lbl_ref)
        top.addWidget(QLabel("DB:"))
        top.addWidget(self.db_input)
        top.addWidget(QLabel("Name:"))
        top.addWidget(self.name_input)
        top.addWidget(QLabel("Tol(Å):"))
        top.addWidget(self.tol_input)
        top.addWidget(self.btn_run)
        top.addWidget(self.progress)
        top.addWidget(self.btn_export)
        top.addWidget(self.btn_open_db)
        layout.addLayout(top)
        
        # New Simulation Controls - dt and steps removed, temperature setter removed
        sim_controls_layout = QHBoxLayout()
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_reset = QPushButton("Reset View")
        self.btn_reset.setEnabled(False) 
        sim_controls_layout.addWidget(self.btn_run)
        sim_controls_layout.addWidget(self.btn_stop)
        sim_controls_layout.addWidget(self.btn_reset)
        sim_controls_layout.addStretch()
        
        layout.addLayout(sim_controls_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout()
        left.setLayout(left_layout)

        self.tabs = QTabWidget()
        self.atoms_table = QTableWidget()
        self.tabs.addTab(self.atoms_table, "Atoms")
        self.bonds_table = QTableWidget()
        self.tabs.addTab(self.bonds_table, "Bonds")
        
        self.summary_box = QTextEdit(); self.summary_box.setReadOnly(True)
        self.tabs.addTab(self.summary_box, "Summary")
        
        # Energy Plot Tab
        self.energy_plot_widget = QWidget()
        self.energy_plot_layout = QVBoxLayout()
        self.energy_plot_widget.setLayout(self.energy_plot_layout)
        self.energy_plot = pg.PlotWidget(title="Energy over time")
        self.energy_plot.addLegend()
        self.ke_curve = self.energy_plot.plot(pen='y', name="Kinetic Energy")
        self.pe_curve = self.energy_plot.plot(pen='g', name="Potential Energy")
        self.total_curve = self.energy_plot.plot(pen='w', name="Total Energy")
        self.energy_plot_layout.addWidget(self.energy_plot)
        self.tabs.addTab(self.energy_plot_widget, "Energy Plot")

        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        self.tabs.addTab(self.log_box, "Log")
        left_layout.addWidget(self.tabs)

        bottom_left = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh DB Summary")
        bottom_left.addWidget(self.btn_refresh)
        self.btn_clear_log = QPushButton("Clear Log")
        bottom_left.addWidget(self.btn_clear_log)
        left_layout.addLayout(bottom_left)

        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout()
        right.setLayout(right_layout)
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 12.0
        self.view.opts['center'] = QVector3D(0.0, 0.0, 0.0)
        try:
            self.view.setBackgroundColor((18, 18, 24, 255))
        except Exception:
            pass
        self.grid_item = gl.GLGridItem()
        self.view.addItem(self.grid_item)
        right_layout.addWidget(self.view)

        # Removed the controls and widgets below the 3D viewer.
        # This includes the "Mode", "Sphere res" widgets and their associated layouts.
        # The color legend will now be rendered manually as part of the GLViewWidget.

        splitter.addWidget(right)
        splitter.setSizes([520, 760])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        self.btn_load.clicked.connect(self.on_load_csv)
        self.btn_ref.clicked.connect(self.on_load_ref)
        self.btn_run.clicked.connect(self.on_run)
        self.btn_stop.clicked.connect(self.on_stop_sim)
        self.btn_export.clicked.connect(self.on_export)
        self.btn_open_db.clicked.connect(self.on_open_db)
        self.btn_refresh.clicked.connect(self.on_refresh_db)
        self.btn_clear_log.clicked.connect(lambda: self.log_box.clear())
        self.btn_reset.clicked.connect(self.on_reset_view)
        
        self._orig_mouse_press = self.view.mousePressEvent
        self._orig_mouse_move = self.view.mouseMoveEvent
        self.view.mousePressEvent = self._mouse_press
        self.view.mouseMoveEvent = self._mouse_move
        
        self._orig_paintGL = self.view.paintGL
        self.view.paintGL = self._paintGL_with_legend
        
        self.atom_positions = None
        self.atom_df = None
        self.selected_atom_idx = -1
        
        self.log("Ready. Load a CSV and start a simulation.")

    def load_initial_csv(self):
        if self.csv_path:
            try:
                df = load_csv(self.csv_path)
                fill_masses(df, mass_col='Mass(amu)')
                for col in ['Vx', 'Vy', 'Vz']:
                    if col not in df.columns:
                        df[col] = 0.0
                return df
            except Exception as e:
                QMessageBox.critical(self, "CSV Error", str(e)[:2000])
                return None
        return None

    @QtCore.pyqtSlot()
    def on_update(self):
        """NEW: Called by a QTimer to update the GUI with the latest simulation data."""
        data = self.data_manager.get_data()
        
        df = data['df']
        bonds = data['bonds']
        ke_sum = data['ke']
        pe_sum = data['pe']
        step_count = data['step']
        
        if df is None or df.empty:
            return
            
        self.time_step_count = step_count
        self.last_df = df
        self.last_bonds = bonds
        
        self.populate_atoms_table(self.last_df, compute_kinetic_energy(self.last_df), compute_coulomb_pe(self.last_df))
        self.populate_bonds_table(self.last_bonds)
        
        self.draw_3d_scene(self.last_df, self.last_bonds)
            
        current_time = self.time_step_count * self.engine.DT_S / 1e-12
        
        self.energy_plot_data['time'].append(current_time)
        self.energy_plot_data['ke'].append(ke_sum)
        self.energy_plot_data['pe'].append(pe_sum)
        self.energy_plot_data['total'].append(ke_sum + pe_sum)
        
        self.ke_curve.setData(self.energy_plot_data['time'], self.energy_plot_data['ke'])
        self.pe_curve.setData(self.energy_plot_data['time'], self.energy_plot_data['pe'])
        self.total_curve.setData(self.energy_plot_data['time'], self.energy_plot_data['total'])
        self.energy_plot.setLabel('left', 'Energy', units='J')
        self.energy_plot.setLabel('bottom', 'Time', units='ps')
        self.energy_plot.setTitle(f"Energy Plot (Step: {self.time_step_count})")
        
        # Max steps can be a user-set parameter, but for this simplified code it's just a large number
        max_steps = 100000 
        self.progress.setValue(int(100 * self.time_step_count / max_steps))

    def log(self, txt: str):
        ts = datetime.now(timezone.utc).isoformat()
        self.log_box.append(f"[{ts}] {txt}")
        print(txt)

    def on_load_csv(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open atom CSV", ".", "CSV Files (*.csv);;All Files (*)")
        if p:
            self.csv_path = p
            df = self.load_initial_csv()
            if df is None: return
            
            self.lbl_csv.setText(os.path.basename(p))
            self.log(f"CSV selected: {p}")
            self.last_df = df
            
            self.last_bonds = detect_bonds(self.last_df, tolerance=self.tolerance)
            self.populate_atoms_table(df, compute_kinetic_energy(df), compute_coulomb_pe(df))
            self.populate_bonds_table(self.last_bonds)
            
            # Reset and draw the initial scene
            self.reset_3d_scene()
            self.draw_3d_scene(df, self.last_bonds, initial=True)

    def on_load_ref(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open reference CSV", ".", "CSV Files (*.csv);;All Files (*)")
        if p:
            self.ref_path = p
            self.lbl_ref.setText(os.path.basename(p))
            self.log(f"Reference selected: {p}")
    
    def on_run(self):
        if not getattr(self, 'csv_path', None):
            QMessageBox.warning(self, "No CSV", "Select a starting CSV first.")
            return
            
        initial_df = self.load_initial_csv()
        if initial_df is None or initial_df.empty:
            QMessageBox.warning(self, "No Atoms Found", "The loaded CSV appears to have no valid atom data to process.")
            return

        self.db_path = self.db_input.text().strip() or "molecular_data.db"
        self.sim_name = self.name_input.text().strip() or os.path.basename(self.csv_path)
        self.tolerance = float(self.tol_input.value())
        
        self.last_bonds = detect_bonds(initial_df, tolerance=self.tolerance)
        if not self.last_bonds:
            QMessageBox.warning(self, "No Bonds", "No bonds detected with the current tolerance. Simulation may not function as expected.")
            
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.progress.setValue(0)
        self.log("Starting continuous simulation...")
        
        self.energy_plot_data = {'time': [], 'ke': [], 'pe': [], 'total': []}
        self.time_step_count = 0
        
        # NEW: Clear and render initial scene
        self.reset_3d_scene()
        self.draw_3d_scene(initial_df, self.last_bonds, initial=True)
        self.last_df = initial_df.copy()

        self.sim_thread = QtCore.QThread()
        self.engine = SimulationEngine(self.last_df, self.last_bonds, self.tolerance, self.data_manager)
        self.engine.moveToThread(self.sim_thread)
        self.sim_thread.started.connect(self.engine.run)
        self.engine.finished.connect(self.on_sim_finished) # NEW: Connect to a finished slot
        self.sim_thread.start()
        
        # NEW: Start the GUI update timer
        self.update_timer.start(50)

    def on_stop_sim(self):
        if self.engine:
            self.engine.stop()
            self.log("Simulation stop requested.")
            # The engine.finished signal will handle clean-up
    
    def on_sim_finished(self):
        """NEW: Slot to handle simulation thread completion."""
        self.sim_thread.quit()
        self.sim_thread.wait()
        self.update_timer.stop() # NEW: Stop the GUI timer
        self.log("Simulation stopped.")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.save_last_frame_to_db()

    def save_last_frame_to_db(self):
        if self.last_df is None: return
        try:
            conn = sqlite3.connect(self.db_path)
            init_db(conn)
            sim_id = save_simulation(conn, self.sim_name, os.path.basename(self.csv_path))
            
            ke = compute_kinetic_energy(self.last_df)
            pe = compute_coulomb_pe(self.last_df)
            save_atoms(conn, sim_id, self.last_df, ke, pe)
            save_bonds(conn, sim_id, self.last_bonds)
            
            rmsd_val = None
            if self.ref_path:
                df_ref = load_csv(self.ref_path)
                fill_masses(df_ref, mass_col='Mass(amu)')
                rmsd_val = compute_rmsd(df_ref, self.last_df)

            total_ke = float(ke.sum())
            total_pe = float(pe.sum() / 2.0)
            avg_bond = float(np.mean([b[2] for b in self.last_bonds]) if self.last_bonds else 0.0)
            
            save_summary(conn, sim_id, len(self.last_df), len(self.last_bonds), total_ke, total_pe, avg_bond, rmsd_val)
            conn.close()
            
            self.last_sim_id = sim_id
            self.log(f"Final frame saved to DB with sim_id={sim_id}")
            summary_text = self.build_summary_text(self.sim_name, self.csv_path, len(self.last_df), len(self.last_bonds),
                                                     total_ke, total_pe, avg_bond, rmsd_val, self.db_path)
            self.summary_box.setPlainText(summary_text)

        except Exception as e:
            self.log(f"DB save error: {e}")

    def on_export(self):
        if self.last_df is None:
            QMessageBox.information(self, "No results", "Run processing first.")
            return
        out = os.path.splitext(self.csv_path)[0] + "_final_boss_summary.txt"
        with open(out, 'w') as f:
            f.write(self.summary_box.toPlainText())
        self.log(f"Summary exported to {out}")
        QMessageBox.information(self, "Saved", f"Summary saved to:\n{out}")

    def on_open_db(self):
        dbp = self.db_input.text().strip()
        if not dbp:
            QMessageBox.warning(self, "DB path", "Set a DB path first.")
            return
        if not os.path.exists(dbp):
            QMessageBox.warning(self, "DB missing", "DB file doesn't exist (run processing to create).")
            return
        folder = os.path.dirname(os.path.abspath(dbp)) or '.'
        try:
            if sys.platform.startswith('win'):
                os.startfile(folder)
            else:
                import subprocess
                if sys.platform.startswith('darwin'):
                    subprocess.run(['open', folder])
                else:
                    subprocess.run(['xdg-open', folder])
        except Exception as e:
            self.log(f"Error opening folder: {e}")

    def on_refresh_db(self):
        dbp = self.db_input.text().strip()
        if not dbp or not os.path.exists(dbp):
            QMessageBox.information(self, "DB not found", "No DB found at path yet.")
            return
        try:
            conn = sqlite3.connect(dbp)
            c = conn.cursor()
            c.execute("SELECT sim_id, name, timestamp, source_file FROM simulations ORDER BY sim_id DESC LIMIT 10")
            sims = c.fetchall()
            txt = "Recent simulations:\n"
            for s in sims:
                txt += f"id:{s[0]} name:{s[1]} file:{s[3]} time:{s[2]}\n"
            self.summary_box.setPlainText(txt)
            conn.close()
            self.log("DB summary refreshed.")
        except Exception as e:
            self.log(f"DB load error: {e}")

    def on_reset_view(self):
        self.on_stop_sim()
        
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_reset.setEnabled(False)
        self.progress.setValue(0)
        self.summary_box.clear()
        
        df = self.load_initial_csv()
        if df is not None:
            self.last_df = df
            self.last_bonds = detect_bonds(self.last_df, tolerance=self.tolerance)
            self.populate_atoms_table(df, compute_kinetic_energy(df), compute_coulomb_pe(df))
            self.populate_bonds_table(self.last_bonds)
            self.reset_3d_scene()
            self.draw_3d_scene(df, self.last_bonds, initial=True)
            self.log("View reset to original molecular structure.")
        else:
            self.log("No CSV file loaded to reset to.")
            QMessageBox.information(self, "No CSV", "No original CSV file is loaded to reset the view.")

    def populate_atoms_table(self, df: pd.DataFrame, ke: pd.Series, pe: pd.Series):
        if df is None or df.empty:
            self.atoms_table.setRowCount(0)
            return
        cols = ['Atom_ID','Element','X','Y','Z','Vx','Vy','Vz','Charge','Mass(amu)','KE (J)','PE (J)']
        self.atoms_table.setColumnCount(len(cols))
        self.atoms_table.setHorizontalHeaderLabels(cols)
        self.atoms_table.setRowCount(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            vals = [
                str(row['Atom_ID']), str(row['Element']),
                f"{row['X']:.6f}", f"{row['Y']:.6f}", f"{row['Z']:.6f}",
                f"{row.get('Vx',0.0)}", f"{row.get('Vy',0.0)}", f"{row.get('Vz',0.0)}",
                f"{row.get('Charge',0.0)}", f"{row.get('Mass(amu)',0.0)}",
                f"{ke.at[row.name]:.6e}",
                f"{pe.at[row.name]:.6e}"
            ]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                self.atoms_table.setItem(i, j, it)
        self.atoms_table.resizeColumnsToContents()

    def populate_bonds_table(self, bonds: List[Tuple[str,str,float]]):
        cols = ['Atom1','Atom2','Length(Å)']
        self.bonds_table.setColumnCount(len(cols))
        self.bonds_table.setHorizontalHeaderLabels(cols)
        self.bonds_table.setRowCount(len(bonds))
        for i, (a1,a2,l) in enumerate(bonds):
            self.bonds_table.setItem(i, 0, QTableWidgetItem(str(a1)))
            self.bonds_table.setItem(i, 1, QTableWidgetItem(str(a2)))
            self.bonds_table.setItem(i, 2, QTableWidgetItem(f"{l:.4f}"))
        self.bonds_table.resizeColumnsToContents()

    def build_summary_text(self, name, csv_path, n_atoms, n_bonds, total_ke, total_pe, avg_bond, rmsd_val, db_path):
        summary = f"""
*** Simulation Summary ***
--------------------------
Simulation Name: {name}
Source File: {os.path.basename(csv_path)}
Time of analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

--- Molecular Properties ---
Number of Atoms: {n_atoms}
Number of Bonds Detected: {n_bonds}
Average Bond Length: {avg_bond:.4f} Å

--- Energy Analysis ---
Total Kinetic Energy: {total_ke:.6e} J
Total Potential Energy (Joule): {total_pe:.6e} J
Total Energy: {total_ke + total_pe:.6e} J

"""
        if rmsd_val is not None:
            summary += f"--- Conformational Analysis ---\nRoot Mean Square Deviation (RMSD): {rmsd_val:.4f} Å\n\n"
        else:
            summary += "--- Conformational Analysis ---\nRMSD not calculated (no reference file provided).\n\n"

        summary += f"--- Database ---"
        summary += f"\nData saved to DB: {db_path}"

        return summary
    
    def reset_3d_scene(self):
        """NEW: A method to clear all dynamic items from the scene."""
        for item in self.sphere_items:
            self.view.removeItem(item)
        for item in self.line_items:
            self.view.removeItem(item)
        self.sphere_items = []
        self.line_items = []
        self.atoms_are_rendered = False
        
    def draw_3d_scene(self, df: pd.DataFrame, bonds: List[Tuple[str,str,float]], initial: bool = False):
        """NEW: This method now either creates or updates the 3D scene."""
        coords = coords_array(df)
        if coords.shape[0] == 0:
            return

        center = coords.mean(axis=0)
        coords_centered = coords - center
        self.atom_positions = coords_centered.copy()
        self.atom_df = df.reset_index(drop=True)

        if initial:
            self.view.reset()
            self.view.addItem(self.grid_item)
            max_span = float(np.max(np.ptp(coords_centered, axis=0))) if coords_centered.size else 1.0
            dist = float(max(8.0, max_span * 2.5))
            self.view.opts['center'] = QVector3D(0.0, 0.0, 0.0)
            self.view.opts['distance'] = dist

        # Hardcoded values for "Ball-and-Stick" view
        sphere_res = 24
        size_scale = 0.5
        covs = np.array([COV_RADII.get(el, COV_RADII.get(el[:2], 0.75)) for el in self.atom_df['Element']])
        sizes = covs * size_scale

        if not self.atoms_are_rendered:
            # Create all items for the first time
            for i in range(len(self.atom_df)):
                pos = self.atom_positions[i]
                el = self.atom_df['Element'].iloc[i]
                md, base_color = make_shaded_sphere_mesh(el, rows=sphere_res, cols=sphere_res)
                mesh = gl.GLMeshItem(meshdata=md, smooth=True, shader='shaded', drawEdges=False)
                radius = float(sizes[i])
                mesh.scale(radius, radius, radius)
                mesh.translate(float(pos[0]), float(pos[1]), float(pos[2]))
                mesh.setGLOptions('opaque')
                mesh.setColor(base_color)
                self.view.addItem(mesh)
                self.sphere_items.append(mesh)

            for a1, a2, length in bonds:
                try:
                    idx1 = self.atom_df.index[self.atom_df['Atom_ID'].astype(str) == str(a1)].tolist()
                    idx2 = self.atom_df.index[self.atom_df['Atom_ID'].astype(str) == str(a2)].tolist()
                    if not idx1 or not idx2:
                        continue
                    p1 = self.atom_positions[idx1[0]]
                    p2 = self.atom_positions[idx2[0]]
                    line = gl.GLLinePlotItem(pos=np.array([p1, p2]), color=(0.9,0.9,0.9,0.85), width=2.0, antialias=True, mode='lines')
                    self.view.addItem(line)
                    self.line_items.append(line)
                except Exception as e:
                    self.log(f"Error drawing bond {a1}-{a2}: {e}")
            self.atoms_are_rendered = True
        else:
            # Update the positions of existing items
            for i, pos in enumerate(self.atom_positions):
                self.sphere_items[i].setTransform(QtGui.QMatrix4x4(1,0,0,pos[0],0,1,0,pos[1],0,0,1,pos[2],0,0,0,1))
            
            # Update bond positions
            for i, bond in enumerate(self.last_bonds):
                a1_id, a2_id, _ = bond
                try:
                    idx1 = self.atom_df.index[self.atom_df['Atom_ID'].astype(str) == str(a1_id)].tolist()
                    idx2 = self.atom_df.index[self.atom_df['Atom_ID'].astype(str) == str(a2_id)].tolist()
                    if not idx1 or not idx2: continue
                    p1 = self.atom_positions[idx1[0]]
                    p2 = self.atom_positions[idx2[0]]
                    self.line_items[i].setData(pos=np.array([p1, p2]))
                except Exception as e:
                    self.log(f"Error updating bond {a1_id}-{a2_id}: {e}")

        self.log(f"3D scene drawn.")

    def _paintGL_with_legend(self, *args, **kwargs):
        self._orig_paintGL()
        
        if self.atom_df is not None:
            painter = QPainter(self.view)
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 10))
            
            elements = sorted(self.atom_df['Element'].unique())
            y_offset = self.view.height() - 20
            x_offset = 10
            
            painter.drawText(x_offset, y_offset, "Color Legend:")
            y_offset -= 15
            
            for el in elements:
                color_tuple = CPK_COLORS.get(el, (0.6, 0.6, 0.6, 1.0))
                color_q = QColor(int(color_tuple[0]*255), int(color_tuple[1]*255), int(color_tuple[2]*255))
                
                painter.setBrush(color_q)
                painter.drawRect(x_offset, y_offset - 8, 10, 10)
                painter.drawText(x_offset + 15, y_offset, el)
                y_offset -= 15
            
            painter.end()


    def _mouse_press(self, ev):
        self._orig_mouse_press(ev)

    def _mouse_move(self, ev):
        self._orig_mouse_move(ev)

    def _highlight_atom(self, idx):
        pass

    def _hover_highlight(self, ev):
        pass

def main():
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()