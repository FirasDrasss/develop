from typing import List, NamedTuple

import pandas as pd


class Position(NamedTuple):
    """Represents a 3D position. Is always the center of a body."""

    x: float
    y: float
    z: float


class Quader:
    """Represents a quader (rectangular prism) with its properties."""

    def __init__(
        self,
        temperature: float,
        position: Position,
        size_x: float,
        size_y: float,
        size_z: float,
    ):
        self.temperature = temperature
        self.position = position
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z

    def __repr__(self) -> str:
        return f"Quader(T={self.temperature}, Pos={self.position}, Size=({self.size_x},{self.size_y},{self.size_z}))"


class Cylinder:
    """Represents a cylinder with its properties."""

    def __init__(
        self, temperature: float, position: Position, radius: float, height: float
    ):
        self.temperature = temperature
        self.position = position
        self.radius = radius
        self.height = height

    def __repr__(self) -> str:
        return f"Cylinder(T={self.temperature}, Pos={self.position}, R={self.radius}, H={self.height})"


class SystemParameters:
    """Holds the parameters for a system of quaders, cylinders, and inflow velocity."""

    def __init__(
        self, quads: List[Quader], cylinders: List[Cylinder], inflow_velocity: float
    ):
        self.quads = quads
        self.cylinders = cylinders
        self.inflow_velocity = inflow_velocity

    def __repr__(self) -> str:
        return f"SystemParameters(V={self.inflow_velocity}, Quads={self.quads}, Cylinders={self.cylinders})"

    @staticmethod
    def from_dataframe_row(row: pd.Series) -> "SystemParameters":
        """
        Creates a SystemParameters object from a pandas DataFrame row.

        The DataFrame row is expected to contain columns as per the original CSV structure:
        T1-T6, x1-x6, y1-y6, xs1-xs4 (quad x-sizes), ys1-ys4 (quad y-sizes),
        zs1-zs4 (quad z-sizes), r5-r6 (cylinder radii), zs5-zs6 (cylinder heights), V.

        Note on Z-Coordinates for Positions:
        This method assumes that z-coordinates for the *positions* of the bodies (e.g., 'z1', 'z2', ..., 'z6')
        are NOT explicitly provided in the input DataFrame row and defaults them to 0.0.
        The fields 'zs1'-'zs4' are interpreted as z-axis *sizes* for quaders,
        and 'zs5'-'zs6' as *heights* for cylinders.
        If your DataFrame includes separate columns for positional z-coordinates (e.g. 'z1', 'z2'),
        you should modify the Position creation logic below to use `float(row[f'z{i}'])`.
        """
        quads: List[Quader] = []
        # Quaders are bodies 1, 2, 3, 4
        for i in range(1, 5):
            pos = Position(
                x=float(row[f"x{i}"]),
                y=float(row[f"y{i}"]),
                z=float(
                    row.get(f"z{i}", 0.0)
                ),  # Assumes z-location 'z{i}' might not exist, defaults to 0.0
            )
            q = Quader(
                temperature=float(row[f"T{i}"]),
                position=pos,
                size_x=float(row[f"xs{i}"]),  # xs1, xs2, xs3, xs4
                size_y=float(row[f"ys{i}"]),  # ys1, ys2, ys3, ys4
                size_z=float(
                    row[f"zs{i}"]
                ),  # zs1, zs2, zs3, zs4 (these are quad z-sizes)
            )
            quads.append(q)

        cylinders: List[Cylinder] = []
        # Cylinders are bodies 5, 6
        for i in range(5, 7):
            pos = Position(
                x=float(row[f"x{i}"]),
                y=float(row[f"y{i}"]),
                z=float(
                    row.get(f"z{i}", 0.0)
                ),  # Assumes z-location 'z{i}' might not exist, defaults to 0.0
            )
            # For cylinders, radius parameters are r5, r6 and height parameters are zs5, zs6
            cyl = Cylinder(
                temperature=float(row[f"T{i}"]),
                position=pos,
                radius=float(row[f"r{i}"]),  # r5, r6
                height=float(row[f"zs{i}"]),  # zs5, zs6 (these are cylinder heights)
            )
            cylinders.append(cyl)

        inflow_velocity = float(row["V"])

        return SystemParameters(
            quads=quads, cylinders=cylinders, inflow_velocity=inflow_velocity
        )


def df_row_to_system_parameters(df: pd.DataFrame, design_id: int) -> SystemParameters:
    """
    Helper function to create a SystemParameters object from a specific row of a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(design_id, int):
        raise TypeError("'design_id' must be an integer.")
    row_id = design_id - 1
    if not 0 <= row_id < len(df):
        raise IndexError(
            f"'row_id' {row_id} is out of bounds for DataFrame with {len(df)} rows."
        )

    row_series = df.iloc[row_id]
    return SystemParameters.from_dataframe_row(row_series)
