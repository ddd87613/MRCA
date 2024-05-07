import pandas as pd
import matplotlib.pyplot as plt
import os


def detect_and_plot_anomalies(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Retrieve all CSV filenames and sort them
    csv_files = [file for file in os.listdir(input_folder) if file.endswith('.csv')]
    csv_files.sort()  # This sorts alphabetically; customize sorting if needed for dates in filenames

    # Process each sorted CSV file
    for filename in csv_files:
        file_path = os.path.join(input_folder, filename)
        data = pd.read_csv(file_path)

        # Prepare a DataFrame for anomaly information
        anomalies_summary = pd.DataFrame()

        # Prepare plotting
        plt.figure(figsize=(10, 6))

        # Metrics to be checked for anomalies
        metrics = ['PodClientLatencyP90(s)', 'PodServerLatencyP90(s)', 'PodClientLatencyP95(s)',
                   'PodServerLatencyP95(s)', 'PodServerLatencyP99(s)']
        colors = ['blue', 'green', 'red', 'cyan', 'magenta']  # Color array for different metrics

        for metric, color in zip(metrics, colors):
            if metric in data.columns:
                mean = data[metric].mean()
                std = data[metric].std()
                cutoff = 3 * std
                lower, upper = mean - cutoff, mean + cutoff

                # Identify anomalies
                data['Anomaly'] = data[metric].apply(lambda x: 'Anomaly' if x < lower or x > upper else 'Normal')

                # Select and append anomalies data
                anomalies = data[data['Anomaly'] == 'Anomaly']
                anomalies_summary = pd.concat([anomalies_summary, anomalies], ignore_index=True)

                # Plot metrics data
                plt.scatter(data.index, data[metric], color=color, label=f'{metric} (Normal)')
                plt.scatter(anomalies.index, anomalies[metric], color='black', marker='x', label=f'{metric} (Anomaly)')

        # Set plot details
        plt.xlabel('Index')
        plt.ylabel('Latencies')
        plt.title('Latency Metrics Anomaly Detection for ' + filename)
        plt.legend()
        plt.grid(True)

        # Save plot to the output folder
        plt.savefig(os.path.join(output_folder, filename.replace('.csv', '_metrics.png')))
        plt.close()

        # Save anomalies summary to the output folder
        anomalies_summary.to_csv(os.path.join(output_folder, filename.replace('.csv', '_anomalies.csv')), index=False)


# Input and output folder paths
input_folder = '8-22 metric'
output_folder = 'output'

# Call the function to process files in the folder
detect_and_plot_anomalies(input_folder, output_folder)
