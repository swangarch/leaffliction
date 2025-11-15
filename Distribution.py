#!/usr/bin/python3

import os
import matplotlib.pyplot as plt


def count_images_in_subdirs(root_dir: str) -> dict:
    """Return a nested dict mapping { plant: { subtype: image_count } }"""
    data = {}
    for plant in os.listdir(root_dir):
        plant_path = os.path.join(root_dir, plant)
        if os.path.isdir(plant_path):
            counts = {}
            for subtype in os.listdir(plant_path):
                subtype_path = os.path.join(plant_path, subtype)
                if os.path.isdir(subtype_path):
                    num_images = sum(
                        1 for f in os.listdir(subtype_path)
                        if os.path.isfile(os.path.join(subtype_path, f))
                    )
                    counts[subtype] = num_images
            if counts:
                data[plant] = counts
    return data


def plot_charts(data: dict) -> None:
    """Generate histogram and pie chart for each plant"""

    colors = ["#D99090", "#90A8D1", "#A8C5A0", "#C2A0D6"]
    for plant, counts in data.items():
        labels = list(counts.keys())
        values = list(counts.values())

        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, values)
        for i, (bar, label) in enumerate(zip(bars, labels)):
            bar.set_label(label)
            bar.set_color(colors[i])
        plt.title(f"{plant} - Image Count by Category")
        plt.xlabel("Category")
        plt.ylabel("Number of Images")
        plt.tight_layout()
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 8))
        plt.pie(values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        plt.legend()
        plt.title(f"{plant} - Category Distribution")
        plt.show()


def main():
    root_dir = os.path.join(".", "leaves")
    if not os.path.isdir(root_dir):
        print(f"Error: directory '{root_dir}' not found.")
        return

    data = count_images_in_subdirs(root_dir)
    if not data:
        print("No valid subdirectories with images found.")
        return

    plot_charts(data)


if __name__ == "__main__":
    main()
