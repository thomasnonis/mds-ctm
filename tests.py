
import os
from zipfile import ZipFile
import subprocess
import sys
def encrypted_code():
	if not os.path.isfile('encrypted.zip'):
		os.system('python -m wget "https://drive.google.com/uc?export=download&id=17I3Vd2mKq_br1SagFvheVZZ0ubzSx9j9" -o encrypted.zip')
		with ZipFile("encrypted.zip", 'r') as zip:
			zip.extractall()
	if sys.platform == 'win32':
		return [subprocess.check_output(['python', 'test.pyc']),subprocess.check_output(['python', 'test.cpython-38.pyc'])]
	else:
		return [subprocess.check_output(['python3', 'test.pyc']),subprocess.check_output(['python3', 'test.cpython-38.pyc'])]

if __name__ == '__main__':
	confirmation = input("Do you really want to run code from {} [y/Y]?\n> ".format("https://drive.google.com/uc?export=download&id=17I3Vd2mKq_br1SagFvheVZZ0ubzSx9j9"))
	if confirmation != "y" and confirmation != "Y":
		sys.exit("Aborted")
	result = "b'Hello World!\n"
	if sys.platform == 'win32':
		result = "b'Hello World!\r\n'"
	try:
		assert all([True for result in encrypted_code() if result == b'Hello World!\r\n']), "Python version should be 3.8!!"
	except subprocess.CalledProcessError:
		print("Python version should be 3.8!! You are running", sys.version )
		sys.exit("Test failed")
	print("All good! You have python3.8 installed.")


