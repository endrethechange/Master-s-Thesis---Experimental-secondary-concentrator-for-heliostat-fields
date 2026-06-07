"""
Autodesk Fusion script for placing heliostat models from JSON data.

This script reads heliostat position and orientation data from a JSON file
located in the same folder as the script. It then inserts heliostat model
files from the active Fusion project and places them in the Fusion design
using the transform matrices stored in the JSON file.

The script supports two heliostat model types:
    - one model for rows 1-8
    - one model for rows 9-14

It also allows the user to choose between the secondary (default) field configuration
and the reference configuration, if both are present in the JSON data.
"""

import traceback
import json
import os
import adsk.core
import adsk.fusion


# Get the active Fusion application and user interface.
# These are used throughout the script for accessing the design,
# prompting the user, inserting components, and showing messages.
app = adsk.core.Application.get()
ui = app.userInterface


# ==========================================================
# SCALING
# ==========================================================

# Scale denominator used for the Fusion model.
#
# Examples:
#     N = 1     -> 1:1 scale
#     N = 10    -> 1:10 scale
#     N = 100   -> 1:100 scale
#     N = 150   -> 1:150 scale
#     N = 200   -> 1:200 scale
#
# The JSON data is assumed to use metres, while Fusion internally works in
# centimetres. Since 1 m = 100 cm, the scale factor becomes 100 / N.
N = 200
SCALE = 100 / N


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def findDataFileByName(folder: adsk.core.DataFolder, name: str):
    """
    Recursively search for a Fusion data file by name.

    Fusion project files are stored inside folders. The required heliostat
    model files may therefore be located directly in the root folder or inside
    one of its subfolders.

    This function searches the given folder first, then searches all subfolders
    until it finds a data file with the requested name.

    Args:
        folder:
            The Fusion data folder to search in.

        name:
            The exact name of the Fusion file to find.

    Returns:
        The matching Fusion DataFile if found, otherwise None.
    """
    # Search files directly inside this folder.
    for df in folder.dataFiles:
        if df.name == name:
            return df

    # Search each subfolder recursively.
    for sub in folder.dataFolders:
        result = findDataFileByName(sub, name)
        if result:
            return result

    # No matching file was found in this folder or its subfolders.
    return None


# ==========================================================
# MAIN RUN FUNCTION
# ==========================================================

def run(_context: str):
    """
    Main Fusion script entry point.

    The script:
        1. Checks that a Fusion design is active.
        2. Finds a JSON file in the same folder as the script.
        3. Reads heliostat position and transform data from the JSON file.
        4. Prompts the user for the Fusion model names to use.
        5. Lets the user choose default or reference configuration data.
        6. Searches the active Fusion project for the required model files.
        7. Inserts and places all heliostats using the JSON transform matrices.
    """
    try:
        # Get the active Fusion product and make sure it is a design.
        design = app.activeProduct

        if not isinstance(design, adsk.fusion.Design):
            ui.messageBox('A Fusion 360 design must be active.')
            return

        # ----------------------------------------------------------
        # JSON file handling
        # ----------------------------------------------------------

        # Locate the folder where this script file is stored.
        # The JSON file is expected to be in the same folder.
        script_dir = os.path.dirname(os.path.realpath(__file__))

        # Find all JSON files in the script folder.
        json_files = [f for f in os.listdir(script_dir) if f.endswith('.json')]

        # Stop if no JSON file is available.
        if not json_files:
            ui.messageBox('No JSON file found in this script’s folder.')
            return

        # Use the first JSON file found in the folder.
        #
        # This assumes that only the intended heliostat layout JSON file is
        # stored next to the script.
        json_path = os.path.join(script_dir, json_files[0])

        # Read the JSON data from disk.
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract the list of heliostats.
        #
        # Each heliostat is expected to contain at least:
        #     - position
        #     - transform
        #
        # Optionally, it may also contain:
        #     - transform_reference
        heliostats = data['heliostats']

        # Build a row index lookup from the distinct Y positions.
        #
        # The script assumes that heliostat rows are separated by their Y
        # coordinate. The sorted Y positions are used to assign row numbers.
        #
        # Rounding is used to avoid small floating-point differences causing
        # two positions from the same row to be treated as different rows.
        row_keys = sorted({round(h['position'][1], 6) for h in heliostats})
        row_index_lookup = {y: idx + 1 for idx, y in enumerate(row_keys)}

        # ----------------------------------------------------------
        # User input
        # ----------------------------------------------------------

        # Ask the user for the Fusion model file used for rows 1-8.
        #
        # The name must match an existing file in the active Fusion project.
        first_rows_name = ui.inputBox(
            'Enter the heliostat model name for rows 1-8 (must exist in your project):'
        )[0].strip()

        # Ask the user for the Fusion model file used for rows 9-14.
        remaining_rows_name = ui.inputBox(
            'Enter the heliostat model name for rows 9-14 (must exist in your project):'
        )[0].strip()

        # Ask whether to use the reference transform data or the default
        # transform data from the JSON file.
        config_choice = ui.inputBox(
            'Use the reference field configuration? Type "yes" to use *_reference data, '
            'or anything else for the default configuration:',
            'Field Configuration',
            'no'
        )[0].strip().lower()

        # Treat several common affirmative inputs as "yes".
        use_reference = config_choice in {'y', 'yes', 'true', '1', 'reference', 'ref'}

        # ----------------------------------------------------------
        # Fusion file lookup
        # ----------------------------------------------------------

        # Search in the active Fusion project for the two heliostat model files.
        active_project = app.data.activeProject
        root_folder = active_project.rootFolder

        # Find the model file used for rows 1-8.
        firstRowsFile = findDataFileByName(root_folder, first_rows_name)

        if not firstRowsFile:
            ui.messageBox(f'No file named "{first_rows_name}" found in project folders.')
            return

        # Find the model file used for rows 9-14.
        remainingRowsFile = findDataFileByName(root_folder, remaining_rows_name)

        if not remainingRowsFile:
            ui.messageBox(f'No file named "{remaining_rows_name}" found in project folders.')
            return

        # Get the root component of the active design.
        #
        # New heliostat occurrences will be inserted into this root component.
        rootComp = design.rootComponent
        occs = rootComp.occurrences

        # ----------------------------------------------------------
        # Place heliostats
        # ----------------------------------------------------------

        for h in heliostats:
            # Read the heliostat position.
            #
            # The Y coordinate is used to determine which row the heliostat
            # belongs to, and therefore which model file should be inserted.
            pos = h['position']

            # Create a new Fusion transform matrix for this heliostat.
            transform = adsk.core.Matrix3D.create()

            # Choose which transform data to use.
            #
            # Default configuration:
            #     transform
            #
            # Reference configuration:
            #     transform_reference
            transform_key = 'transform_reference' if use_reference else 'transform'
            transform_data = h.get(transform_key)

            # Validate that the transform data has the expected 3 x 4 format.
            #
            # The first three columns describe orientation.
            # The fourth column describes translation.
            if (
                not transform_data
                or len(transform_data) != 3
                or any(len(row) != 4 for row in transform_data)
            ):
                ui.messageBox(
                    f'Missing or invalid {transform_key} data in JSON; cannot place heliostats.'
                )
                return

            # Copy the 3 x 3 orientation part of the transform matrix.
            #
            # This preserves the heliostat orientation from the source data.
            for row in range(3):
                for col in range(3):
                    transform.setCell(row, col, transform_data[row][col])

            # Set the translation part of the transform.
            #
            # The translation values are scaled from JSON units to the Fusion
            # model scale.
            transform.translation = adsk.core.Vector3D.create(
                transform_data[0][3] * SCALE,
                transform_data[1][3] * SCALE,
                transform_data[2][3] * SCALE
            )

            # Determine the heliostat row based on its Y position.
            row_key = round(pos[1], 6)
            row_index = row_index_lookup.get(row_key)

            # Use one model for rows 1-8 and another model for rows 9-14.
            target_file = firstRowsFile if row_index and row_index <= 8 else remainingRowsFile

            # Insert the selected heliostat model into the design using the
            # transform matrix from the JSON file.
            occs.addByInsert(target_file, transform, False)

        # Show a summary when all heliostats have been inserted.
        ui.messageBox(
            f'Placed {len(heliostats)} heliostats using "{os.path.basename(json_path)}" '
            f'({first_rows_name} for rows 1-8, {remaining_rows_name} for rows 9-14) '
            f'with {"reference" if use_reference else "default"} configuration.'
        )

    except Exception:
        # Log the full Python traceback to Fusion's text command/log output.
        # This makes debugging easier if the script fails.
        app.log(f'Failed:\n{traceback.format_exc()}')
