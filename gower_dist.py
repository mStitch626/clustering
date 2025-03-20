import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
def get_feature_types(column):
    if all(isinstance(x, (int, float)) for x in column):
        return 'numerical'
    return 'categorical'

def normalize_numerical_features(data):
    num_cols = [i for i in range(data.shape[1]) if get_feature_types(data[:, i]) == 'numerical']
    
    if num_cols:
        scaler = MinMaxScaler()
        data[:, num_cols] = scaler.fit_transform(data[:, num_cols])
    
    return data, num_cols

def numerical_distance(xi, xj):
    return abs(xi - xj)

def categorical_distance(xi, xj):
    return 0 if xi == xj else 1

def gower_distance(xi, xj, num_cols):
    """Compute Gower distance between two rows."""
    p = len(xi)
    dissimilarities = np.zeros(p)

    for m in range(p):
        if m in num_cols:  # Numerical feature
            dissimilarities[m] = numerical_distance(xi[m], xj[m])
        else:  # Categorical feature
            dissimilarities[m] = categorical_distance(xi[m], xj[m])
    
    return np.mean(dissimilarities)

def gower_distance_matrix(data):
    data = np.array(data, dtype=object)
    data, num_cols = normalize_numerical_features(data)

    n = len(data)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = gower_distance(data[i], data[j], num_cols)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
            
    return distance_matrix


# Example 

df = pd.read_csv("dataset/kdd_data/KDDTrain+.txt", header=None)
columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot'
,'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations'
,'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate'
,'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count'
,'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate'
,'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','outcome','level'])
df.columns = columns


x = df.iloc[1:5,:] ### only 5 data rows

print(gower_distance_matrix(x))