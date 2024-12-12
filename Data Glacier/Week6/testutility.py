import yaml
import pandas as pd

# Function to read the YAML file
def read_config_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error reading YAML file: {exc}")
        return None

# Function to validate column headers
def column_header(df_result, table_config):
    df_result.columns = df_result.columns.str.strip()
    df_result.columns = df_result.columns.str.lower()  # Ensures case-insensitivity
    
    # Removing special characters with underscore
    df_result.columns = df_result.columns.str.replace(r'\W+', '_', regex=True)

    # Extract expected columns from YAML
    expected_columns = list(table_config['columns'])
    expected_columns.sort()

    # Sort actual columns for comparison
    actual_columns = list(df_result.columns)
    actual_columns.sort()

    # Compare sorted columns
    if actual_columns == expected_columns:
        print("Column name and column length validation passed")
        return 1
    else:
        print("Column name and column length validation failed")
        mismatched_columns = set(actual_columns) - set(expected_columns)
        missing_in_yaml = set(expected_columns) - set(actual_columns)
        print("The following columns are not in the YAML file:", list(mismatched_columns))
        print("The following YAML columns are not in the file uploaded:", list(missing_in_yaml))
        return 0
