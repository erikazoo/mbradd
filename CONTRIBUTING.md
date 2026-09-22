# Contributing to Medicare Billing Risk & Anomaly Detection Dashboard

Fresh machine to running notebook, picking up from the setup within branch: *data-preprocessing-erica*.  
Tools: Python 3.12, managed w/ uv, notebook in marimo. Run via VS Code.  
Install on machine:
- Git for Windows
- VS Code + marimo extension
- uv: Python package and env manager   
___
**uv from windows:**
```powershell
winget install --id=astral-sh.uv -e
```
in a new terminal/restart VS Code: 
```powershell
uv --version
```
**uv from linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # or: brew install uv
```
___
___
## 1. Clone this repository
```bash 
git clone https://github.com/erikazoo/mbradd
cd mbradd
```
## 2. (Windows) Allow venv to activate 
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
## 3. Create environment 
```bash
uv venv --python 3.12
```
Activate it:
- **Windows:** `.venv\Scripts\activate`
- **macOS / Linux:** `source .venv/bin/activate`
```bash
uv pip install -r requirements.txt # DOES NOT EXIST YET -- skip to directions below.

uv pip install marimo pandas numpy scikit-learn matplotlib seaborn
```

## 4. Point VS Code to environment
Extensions sidebar > Search 'marimo' > install verified one  
'ctrl/cmd+shift+p' (Command palatte) > Python: Select Interpreter > '.venv'  
*note* kernel and interpreter must be the same. 

## 5. Data loading
CMS dataset is not in the repo (.gitignore), as it is redownloadable. Opening 'data_preprocessing_ez.py' in the marimo notebook view and running the first cell will download the file into 'data/raw/' (guarded, meaning it will be skipped in later runs so data isn't constantly being reinitialized). 

## To be updated. 