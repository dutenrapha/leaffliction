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
    for root, _, files in os.walk(directory):
        if files:  # Only include folders that contain files
            relative_path = os.path.relpath(root, directory)
            folder_counts[relative_path] = len(files)
    return folder_counts


def create_charts(folder_counts):
    """
    Create a pie chart and a bar chart displaying the
    distribution of files across folders.

    :param folder_counts: A dictionary containing the counts
    of files in each folder.
    """
    # Data for the charts
    labels = list(folder_counts.keys())
    sizes = list(folder_counts.values())

    colors = plt.cm.tab10.colors  # Use the 'tab10' colormap
    color_map = {
        label: colors[i % len(colors)]
        for i, label in enumerate(labels)
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(wspace=0.4)

    # Pie chart
    ax1.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=[color_map[label] for label in labels]
    )
    ax1.axis('equal')  # Equal aspect ratio ensures
    ax1.set_title('File Distribution by Folder')

    # Bar chart
    ax2.bar(labels, sizes, color=[color_map[label] for label in labels])
    ax2.set_xlabel('Folders')
    ax2.set_ylabel('Number of Files')
    ax2.set_title('Number of Files per Folder')
    ax2.tick_params(axis='x', rotation=60)

    plt.tight_layout()
    plt.savefig('output.png')
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

    folder_counts = count_files_in_subfolders(directory)
    create_charts(folder_counts)
