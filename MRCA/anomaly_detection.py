import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import pytz
import os

import os
import os
import pandas as pd


def process_files(input_folder1, input_folder2, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the first input folder
    for filename1 in os.listdir(input_folder1):
        if filename1.endswith('.csv'):
            file1 = os.path.join(input_folder1, filename1)

            # Search for matching file in the second input folder
            for filename2 in os.listdir(input_folder2):
                if filename2.endswith('.csv') and filename1[:10] == filename2[:10]:
                    file2 = os.path.join(input_folder2, filename2)
                    output_file = os.path.join(output_folder, filename1)

                    # Load the files
                    data_1 = pd.read_csv(file1)
                    data_2 = pd.read_csv(file2)

                    # Convert the timestamp in data_1 to Unix timestamp
                    data_1['Timestamp'] = pd.to_datetime(data_1['Timestamp']).astype(int) / 10 ** 9

                    # Convert the Unix timestamp in data_2 to datetime (only use first 10 digits)
                    data_2['StartTimeUnixNano'] = pd.to_datetime(data_2['StartTimeUnixNano'].astype(str).str[:10],
                                                                 unit='s')

                    # Create a new column in data_1 to hold the Duration values
                    data_1['Duration'] = 0

                    # Insert Duration values from data_2 to the closest timestamps in data_1
                    for i, row in data_2.iterrows():
                        closest_idx = (data_1['Timestamp'] - row['StartTimeUnixNano'].timestamp()).abs().idxmin()
                        data_1.at[closest_idx, 'Duration'] = row['Duration']

                    # Fill any NaN values with 0 in data_1
                    data_1['Duration'].fillna(0, inplace=True)

                    # Rename columns in data_1
                    new_columns = ['Timestamp'] + [str(i) for i in range(1, len(data_1.columns))]
                    data_1.columns = new_columns

                    # Save the modified data_1 to a new CSV file
                    data_1.to_csv(output_file, index=False)
                    print(f"Processed file saved as {output_file}")



# Define folder paths
input_folder1 = 'processed_data/normal_data/log_template'
input_folder2 = 'processed_data/normal_data/trace_latency'
output_folder = 'processed_data/normal_data/aggregation'

process_files(input_folder1,input_folder2,output_folder)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.mu = nn.Linear(hidden_size, latent_size)
        self.sigma = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        x = torch.relu(self.linear(x))
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x


class VAE(nn.Module):
    def __init__(self, input_size, output_size, latent_size, hidden_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, output_size)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        return self.decoder(z), mu, sigma


def load_injection_times(file_path):
    with open(file_path, 'r') as file:
        times = [line.strip() for line in file if line.strip()]
    return times


def train_vae(input_folder, model_save_path, epochs=10000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(input_size=1, output_size=1, latent_size=16, hidden_size=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            log_path = os.path.join(input_folder, filename)
            df = pd.read_csv(log_path)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)

            # Assuming normal data is available for training
            data_column = df.columns[1]
            data = df[data_column].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            data_tensor = torch.tensor(scaled_data, dtype=torch.float32).to(device)

            for epoch in range(epochs):
                optimizer.zero_grad()
                reconstructed, _, _ = model(data_tensor)
                loss = torch.nn.functional.mse_loss(reconstructed, data_tensor)
                loss.backward()
                optimizer.step()

    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')


def detect_anomalies(input_folder, output_folder, injection_times, model_path, threshold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(input_size=1, output_size=1, latent_size=16, hidden_size=128).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    os.makedirs(output_folder, exist_ok=True)

    for target_time_str in injection_times:
        mse_scores = {}
        target_time = datetime.strptime(target_time_str, '%Y-%m-%d %H:%M:%S')
        target_time = pytz.utc.localize(target_time)
        start_time = target_time - timedelta(minutes=5)
        end_time = target_time + timedelta(minutes=5)

        for filename in os.listdir(input_folder):
            if filename.endswith('.csv'):
                log_path = os.path.join(input_folder, filename)
                df = pd.read_csv(log_path)
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
                filtered_df = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] <= end_time)]

                if filtered_df.empty:
                    continue

                data_column = filtered_df.columns[1]
                data = filtered_df[data_column].values.reshape(-1, 1)
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data)
                data_tensor = torch.tensor(scaled_data, dtype=torch.float32).to(device)

                with torch.no_grad():
                    reconstructed, _, _ = model(data_tensor)
                    mse_loss = torch.nn.functional.mse_loss(reconstructed, data_tensor).item()
                    mse_scores[filename] = mse_loss

        sorted_services = sorted(mse_scores.items(), key=lambda x: x[1])

        # Save sorted services for this timestamp
        result_file = os.path.join(output_folder, f'ranked_services_{target_time_str.replace(":", "-")}.csv')
        with open(result_file, 'w') as f:
            for service, mse in sorted_services:
                is_anomalous = mse > threshold
                f.write(f"{service},{mse},{is_anomalous}\n")


train_input_folder = 'processed_data/normal_data/aggregation'
model_save_path = 'vae_model.pth'
train_vae(train_input_folder, model_save_path)

detect_input_folder = 'processed_data/abnormal_data/aggregation'
detect_output_folder = 'anomaly_detection/anomaly_score'
time_file = 'anomaly_detection/suffer_anomaly_inject_data_1/inject anomaly time point.txt'
threshold = 0.01


injection_times = load_injection_times(time_file)
detect_anomalies(detect_input_folder, detect_output_folder, injection_times, model_save_path, threshold)
