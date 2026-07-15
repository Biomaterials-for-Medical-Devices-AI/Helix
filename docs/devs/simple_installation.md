# Installing Helix for non-experience users:

**Heads up:** This guide uses Visual Code 

 When using Visual Studio Code clone Helix data from the Github repo using the option: Clone git repository

 If that option is not available you might have to download git from: https://git-scm.com/install/windows?utm_source=chatgpt.com close and restart visual code after

 Once git is installed you should be able to clone the Git repository from: https://github.com/Biomaterials-for-Medical-Devices-AI/Helix

 Create a new environment using: 
 ```shell
 python -m env. 
 ```

 if error is found that says: "Python was not found" try: 
 ```shell
 python --version or py --version
 ```

 create a new environment using:
 ```shell
 py_m venv. venv 
 ```
 # or
 ```shell
 pyton_m venv. 
 ```
---

## Next step is to activate the virtual environment using 
 ```shell
 .venv/Scripts/activate 
 ```

## if error reading: " cannot be loaded because running scripts is disabled on this system" then try:
 ```shell
 Set-ExecutionPolicy_Scope CurrentUser RemoteSigned 
 ```
 after that delete terminal then type:
 ```shell
 .venv\Scripts\activate
 ```
# if succesful you should see: (.venv) 

# now install uv using:
 ```shell
 pip install uv 
 ```
#  write the following on the terminal:
 ```shell    
 uv sync --all-groups 
 ```

## if you encounter an error such as : "failed to remove directory"  then try typing: 
 ```shell
 deactivate
 ```

# after that: 
 ```shell
 uv sync --all-groups
 ```
     
# if that does not work try:
 ```shell
 winget install --id=Astral-sh.uv
 ```
# and try typing in the terminal: 
 ```shell
 python -m pip install --user uv or py-m pip install --user uv
 ```

## if you are still getting an error like: "No module names uv" you will have to uninstall, install it again and delete the terminal.

# To uninstall, first type on terminal:
 ```shell
 winget uninstall astral-sh.uv 
 ```
# after that type : 
 ```shell
 powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
 ```
# delete terminal

# open terminal again and type:
 ```shell
 uv --version 
 ```

## if the following error appears: The term 'uv' is not recognized as the name of an endlet, function, etc.." then type:
# make sure you type YOUR OWN directory name in the template:
```shell
 dir C:\Users\("your directory name")\.local\bin
 ```

# if a further error is encounter it means we have to tell the system where we installed UV as it cannot find it

# To do this manually follow the instructions underneath:

# Press Win and search for “Edit the system environment variables”.
# Click Environment Variables…
# Under User variables for ameli, select Path.
# Click Edit.
# Click New.
# Add: C:\Users\("your directory name")\.local\bin

# once completed close visual code and any other terminal that is opened

# open visual code again and type:
 ```shell
 uv --version
 ```
# after that type on the terminal: 
 ```shell
 uv sync --all-groups
 ```
# uv will install all packages needed. Allow a couple of minutes until completed

# To test all is working correctly, type: 
 ```shell
 uv run helix
 ```

# Allow a couple of minutes for helix to load

# A new page will open on your internet browser with Helix

# Congrats! Now you can use helix! 
