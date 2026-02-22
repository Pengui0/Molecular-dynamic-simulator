What the simulator does...
The Molecular Dynamics Simulator performs real-time simulation and visualization of molecular systems. 
It calculates atomic motion using classical mechanics and updates particle positions using the Velocity Verlet integration algorithm. 
The simulator models bonded and non-bonded interactions such as Morse potential, harmonic angle potential, and Coulombic forces. 
It also maintains temperature stability using a Berendsen thermostat and provides real-time energy tracking, RMSD analysis, and 3D visualization of molecular motion.

Libraries used...
Python 3.10+
NumPy for numerical computation and vectorized physics calculations
Pandas for structured molecular data handling
PyQt6 for GUI development
PyQtGraph (OpenGL module) for real-time 3D rendering
SQLite for persistent simulation data storage
PyOpenGL for graphical rendering support


What concepts does it demonstrate...
Classical Mechanics (Newton’s equations of motion)
Velocity Verlet integration algorithm
Molecular force field modeling (Morse potential, harmonic potential, Coulomb interaction)
Thermodynamic control using Berendsen thermostat (NVT ensemble)
Multithreading in GUI applications (QThread separation of the simulation engine)
Real-time OpenGL 3D rendering
Data persistence using SQLite
RMSD calculation using the Kabsch algorithm
