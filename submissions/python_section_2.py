import pandas as pd
import datetime

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    toll_ids = pd.concat([df['id_start'], df['id_end']]).unique()
    
    dist_matrix = pd.DataFrame(np.inf, index=toll_ids, columns=toll_ids)
    
    np.fill_diagonal(dist_matrix.values, 0)
    
    for _, row in df.iterrows():
        id_start, id_end, distance = row['id_start'], row['id_end'], row['distance']
        dist_matrix.loc[id_start, id_end] = distance
        dist_matrix.loc[id_end, id_start] = distance  

    for k in toll_ids:
        for i in toll_ids:
            for j in toll_ids:
                dist_matrix.loc[i, j] = min(dist_matrix.loc[i, j], dist_matrix.loc[i, k] + dist_matrix.loc[k, j])
    
    return dist_matrix


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    unrolled_data = []
    
    ids = df.index
    
    for i, id_start in enumerate(ids):
        for j, id_end in enumerate(ids):
            if i != j:  
                distance = df.at[id_start, id_end]
                unrolled_data.append([id_start, id_end, distance])
    
    unrolled_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])
    
    return unrolled_df

def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    
    lower_bound = ref_avg_distance * 0.9
    upper_bound = ref_avg_distance * 1.1

    avg_distances = df.groupby('id_start')['distance'].mean().reset_index()

    valid_ids = avg_distances[(avg_distances['distance'] >= lower_bound) & (avg_distances['distance'] <= upper_bound)]['id_start']

    return df[df['id_start'].isin(valid_ids)]


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    moto_rate = 0.8
    car_rate = 1.2
    rv_rate = 1.5
    bus_rate = 2.2
    truck_rate = 3.6
    
    # Calculate toll rates by multiplying the distance with the respective coefficients
    df['moto'] = df['distance'] * moto_rate
    df['car'] = df['distance'] * car_rate
    df['rv'] = df['distance'] * rv_rate
    df['bus'] = df['distance'] * bus_rate
    df['truck'] = df['distance'] * truck_rate

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
        time_intervals = [
        (datetime.time(0, 0, 0), datetime.time(10, 0, 0), 'Friday'),  
        (datetime.time(10, 0, 0), datetime.time(18, 0, 0), 'Saturday'),
        (datetime.time(18, 0, 0), datetime.time(23, 59, 59), 'Sunday'),
    ]
    
    weekend_interval = (datetime.time(0, 0, 0), datetime.time(23, 59, 59), 'Sunday')  

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    weekday_discount_factors = [0.8, 1.2, 0.8]
    weekend_discount_factor = 0.7

    new_rows = []

    for _, row in df.iterrows():
        for i, (start_time, end_time, end_day) in enumerate(time_intervals):
            new_row = row.copy()
            new_row['start_day'] = days[i]  
            new_row['start_time'] = start_time
            new_row['end_day'] = end_day    
            new_row['end_time'] = end_time
            
            discount_factor = weekday_discount_factors[i]
            new_row['moto'] *= discount_factor
            new_row['car'] *= discount_factor
            new_row['rv'] *= discount_factor
            new_row['bus'] *= discount_factor
            new_row['truck'] *= discount_factor

            new_rows.append(new_row)

        new_row = row.copy()
        new_row['start_day'] = 'Saturday'
        new_row['start_time'] = weekend_interval[0]
        new_row['end_day'] = weekend_interval[2]
        new_row['end_time'] = weekend_interval[1]
        
        new_row['moto'] *= weekend_discount_factor
        new_row['car'] *= weekend_discount_factor
        new_row['rv'] *= weekend_discount_factor
        new_row['bus'] *= weekend_discount_factor
        new_row['truck'] *= weekend_discount_factor

        new_rows.append(new_row)
    
    new_df = pd.DataFrame(new_rows)
    
    return new_df
