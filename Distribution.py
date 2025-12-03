#!/usr/bin/python3

import os
import matplotlib.pyplot as plt
import sys


def count_images_in_subdirs(root_dir: str) -> dict:
    """Return a nested dict mapping { plant: { subtype: image_count } }"""
    data = {}
    for subtype in os.listdir(root_dir):
        subtype_path = os.path.join(root_dir, subtype)
        if os.path.isdir(subtype_path):
            num_images = sum(
                1 for f in os.listdir(subtype_path)
                if os.path.isfile(os.path.join(subtype_path, f))
            )
        data[subtype] = num_images
    return data


def plot_charts(data: dict, title: str) -> None:
    """Generate histogram and pie chart for each plant"""

    colors = ["#D99090", "#90A8D1", "#A8C5A0", "#C2A0D6",
              "#8CBAC5", "#E5BE86", "#D27B7B", "#AE71AD"]

    plant = data.keys()
    counts = data.values()
    labels = list(plant)
    values = list(counts)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)
    for i, (bar, label) in enumerate(zip(bars, labels)):
        bar.set_label(label)
        bar.set_color(colors[i])
    plt.title(f"{title} - Image Count by Category")
    plt.xlabel("Category")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90)
    plt.legend()
    plt.title(f"{title} - Category Distribution")
    plt.show()


def main():
    try:
        if len(sys.argv) != 2:
            raise TypeError("Wrong number of arguments. " +
                            "Usage: python Distribution.py <path>")
        root_dir = os.path.join(".", sys.argv[1])
        if not os.path.isdir(root_dir):
            print(f"Error: directory '{root_dir}' not found.")
            return
        data = count_images_in_subdirs(root_dir)
        if not data:
            print("No valid subdirectories with images found.")
            return
        plot_charts(data, sys.argv[1])
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
