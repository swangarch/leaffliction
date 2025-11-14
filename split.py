import os
import sys
import shutil
import random as rd


def move_file(src, dst, file):
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
				count = 0
				for file in os.listdir(os.path.join(path, subdir)):
					if rd.random() > 0.9:
						move_file(os.path.join(path, subdir), os.path.join(path_test, subdir), file)
						count += 1
					if count > 10:
						break
	except Exception as e:
		print("Error:", e)


if __name__ == "__main__":
	main()
