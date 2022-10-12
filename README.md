## Jump in the environment

### Ubuntu

```sh
sudo apt install python3.8
sudo apt install python3-virtualenv
virtualenv -p $(which python3.8) venv
source venv/bin/activate
pip install -r requirements.txt
```


### Windows
Download python3.8 from https://www.python.org/downloads/ and follow instruction.
```PowerShell
pip install virtualenv
virtualenv -p "C:\Users\<USERNAME>\AppData\Local\Programs\Python\Python38\python.exe" venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\activate
pip install -r requirements.txt
```