from typing import Dict, List

import pandas as pd
import itertools
import re
import polyline
from geopy.distance import geodesic

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    lst1 = lst
    lst = []
    while (lst1 != []):
        sub = lst1[0:n]
        for j in sub[::-1]:
            lst.append(j)
        del lst1[0:n]
    return lst
    



def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here

    dict = {}  
    for s in lst:
        length = len(s)
        if length not in dict:
            dict[length] = []
        dict[length].append(s)
    dict = {k: dict[k] for k in sorted(dict)}
    return dict
    
def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    def _flatten_dict(nested_dict, parent_key):
        items = []

        for key, value in nested_dict.items():
            new_key = parent_key + sep + key if parent_key else key

            if isinstance(value, dict):
                items.extend(_flatten_dict(value, new_key).items())
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    items.extend(_flatten_dict({f'[{i}]': item}, new_key).items())
            else:
                items.append((new_key, value))

        return dict(items)

    return _flatten_dict(nested_dict, '')


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    all_perm = itertools.permutations(nums)
    unique_perm = set(all_perm)
    return(list(unique_perm))


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    pattern = r"\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2}|\d{2}-\d{2}-\d{4}"
    match_list = re.findall(pattern,text)
    return match_list

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    decoded_lst = polyline.decode(polyline_str)
    data = pd.DataFrame(decode_lst, columns = ['latitude','longitude'])
    data['distance'] = 0.0
    for i in range(1, len(data)):
        prev_coords = (data.loc[i-1, 'latitude'], data.loc[i-1, 'longitude'])
        current_coords = (data.loc[i, 'latitude'], data.loc[i, 'longitude'])
    
    # Calculate the geodesic distance using the Haversine formula
        data.loc[i, 'distance'] = geodesic(prev_coords, current_coords).kilometers
    
    return data



def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    transposed_matrix = list(zip(*matrix))
    rotated_matrix = [list(row[::-1]) for row in transposed_matrix]
    n = len(rotated_matrix)
    result_matrix = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            result_matrix[i][j] = row_sum + col_sum
    return result_matrix


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    day_mapping = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    
    df['start_day_num'] = df['startDay'].map(day_mapping)
    df['end_day_num'] = df['endDay'].map(day_mapping)

    df['start_time'] = pd.to_datetime(df['startTime']).dt.time
    df['end_time'] = pd.to_datetime(df['endTime']).dt.time

    grouped = df.groupby(['id', 'id_2'])

    def check_complete_timestamps(group):
        unique_days = group['start_day_num'].unique()
        
        if len(unique_days) < 7:
            return True
        
        hours_covered = {day: set() for day in unique_days}

        for _, row in group.iterrows():
            start_hour = pd.Timestamp.combine(pd.Timestamp.min, row['start_time']).hour
            end_hour = pd.Timestamp.combine(pd.Timestamp.min, row['end_time']).hour

            for hour in range(start_hour, end_hour + 1):
                hours_covered[row['start_day_num']].add(hour)

        for day, hours in hours_covered.items():
            if len(hours) < 24:
                return True  

        return False  

    result = grouped.apply(check_complete_timestamps)
    return result.astype(bool)

    return pd.Series()
