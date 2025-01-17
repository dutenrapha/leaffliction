import os
import sys
import matplotlib.pyplot as plt

def count_files_in_subfolders(directory):
    """
    Count the number of files in each subfolder of the specified directory.

    :param directory: Path to the directory to scan.
    :return: A dictionary with folder names as keys and file count as values.
    """
    folder_counts = {}
    for entry in os.scandir(directory):
        if entry.is_dir():  # Check if the entry is a directory
            folder_counts[entry.name] = len(os.listdir(entry.path))
    return folder_counts

def create_charts(folder_counts):
    """
    Create a pie chart and a bar chart displaying the distribution of files across folders.

    :param folder_counts: A dictionary containing the counts of files in each folder.
    """
    # Data for the charts
    labels = list(folder_counts.keys())
    sizes = list(folder_counts.values())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Create subplots for the two charts
    
    # Pie chart
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title('File Distribution by Folder')
    
    # Bar chart
    ax2.bar(labels, sizes, color='blue')
    ax2.set_xlabel('Folders')
    ax2.set_ylabel('Number of Files')
    ax2.set_title('Number of Files per Folder')
    ax2.tick_params(axis='x', rotation=45)  # Rotate x labels for better visibility
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Check if the directory path is provided as an argument
    if len(sys.argv) != 2:
        print("Usage: python Distribution.py [path_to_directory]")
        sys.exit(1)

    directory = sys.argv[1]
    # Check if the directory exists
    if not os.path.exists(directory):
        print("Error: The directory does not exist")
        sys.exit(1)

    folder_counts = count_files_in_subfolders(directory)  # Count files in subfolders
    print(folder_counts)
    print(max(folder_counts.values()))
    create_charts(folder_counts)  # Generate and display charts
