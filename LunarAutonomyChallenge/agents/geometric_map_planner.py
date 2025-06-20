import numpy as np
from leaderboard.agents.geometric_map import GeometricMap

class GeometricMapPlanner:
    def __init__(self, geometric_map, init_pos, init_orient):
        """Initialize with GeometricMap instance
        Args: geometric_map: GeometricMap - The geometric map object to manage
        """
        
        self.map = geometric_map
        self.initial_map_array = self.get_map_array()
        self.map_size = self._get_map_size()
        self.cell_size = self._get_cell_size()

        self.mapping_threshold = self.cell_size * 0.5

        # Create a dictionary to store points by cell
        self.cell_heights = {}
        self.cell_rocks = {}

        # Exploration tracking 
        self.exploration_grid = np.zeros(
                self.initial_map_array.shape[:2], dtype=bool)
        self.current_exploration_cell = (0, 0)
        self.exploration_path = []
        self.exploration_index = 0
        self.exploration_complete = False

        # Generate initial exploration path
        self._generate_exploration_path()
        self._set_initial_map_array(init_pos, init_orient)

        print("[MAP] Initialized geometric map manager")
        print(f"[MAP] Initial map array shape: {self.initial_map_arra.shape}")
        print(f"[MAP] Map size: {self.map_size} meters")
        print(f"[MAP] Cell size: {self.cell_size} meters")
        print(f"[MAP] Genarated exploration path with {len(self.exploration_path)} cells")

    def _set_initial_map_array(self, init_pos, init_orient):
        for i in range(self.initial_map_array.shape[0]):
            for j in range(self.initial_map_array.shape[1]):
                # Add random normal distribution with std dev of 0.3
                random_offset = np.random.normal(0, 0.3)
                # 50% chance to set rock flag to True
                is_rock = np.random.random() < 0.6
                self.update_cell_terrain(
                        (i, j), init_pos[2] + 0.134 + random_offset, is_rock)
                print(f"[MAP] Done setting initial map array")

    def _generate_exploration_path(self):
        """Generate a path to visit all cells in the map using a serpentine pattern"""
        rows, cols = self.initial_map_array.shape[:2]
        self.exploration_path = []

        # Generate serpentine pattern
        for i in range(rows):
            rows_cells = []
            for j in range(cols):
                row_cells.append((i, j))

                # Reverse every other row for more effcient traversal
                if i % 2 == 1:
                    row_cells.reverse()

                self.exploration_path.extend(row_cells)

    def get_next_exploration_target(self):
        """ Get the next cell to explore
        Returns:
        tuple: (x, y) world coordinates of the next cell to explore, or None if exploration is completed
        """
        if self.exploration_complete:
            print("[MAP] Exploration already comlete")
            return None

        if self.exploation_index >= len(self.exploration_path):
            self.exploration_complete = True
            print("[MAP] Exploration complete")
            return None

        # Get next cell indices
        next_cell = self.exploration_path[self.exploration_index]
        self.current_exploration_cell = next_cell

        # Conver to world coordinates
        cell_data = self.get_cell_position(*next_cell)
        if cell_data is None:
            # Skip invalid cells
            self.exploration_index += 1
            return self.get_next_exploration_target()

        # Mark as visited in our tracking grid
        self.exploration_grid[next_cell] = True

        # Get world coordinates of cell center
        cell_coords = cell_data

        print(f"[MAP] Next exploration target: Cell {next_cell} at world coordinates ({cell_coords[0]:.2f}, {cell_coords[1]:.2f})")
        return cell_coords

    

