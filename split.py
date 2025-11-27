#!/usr/bin/python3

import os
import sys
import shutil
import random as rd


def move_file(src: str, dst: str, file: str) -> None:
	"""Move a file from a source directory to the other.
	file argument is the filename."""
	try:
		f_src = os.path.join(src, file)
		if not os.path.exists(dst):
			os.makedirs(dst, exist_ok=True)
		f_dst = os.path.join(dst, file)
		shutil.move(f_src, f_dst)
	except Exception as e:
		print("Error:", e)


def main():
	try:
		if len(sys.argv) == 2:
			path = sys.argv[1]
			path_test = path + "_test" 
			os.makedirs(path_test, exist_ok=True)
			for subdir in os.listdir(path):
				files_in_subdir = os.listdir(os.path.join(path, subdir))
				sample = rd.sample(files_in_subdir, 100)
				for file in sample:
					move_file(os.path.join(path, subdir), 
			   				  os.path.join(path_test, subdir), 
							  file)
		else:
			raise TypeError("Wrong argument number, Usage: python split.py <path>")
	except Exception as e:
		print("Error:", e)


if __name__ == "__main__":
	main()
