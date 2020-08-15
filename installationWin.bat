@echo off
cmd /k "python -m pip install --upgrade pip & pip install virtualenv & virtualenv env & .\env\Scripts\activate & pip install -r requirements.txt && exit"