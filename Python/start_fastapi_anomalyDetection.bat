@echo off
cd /d C:\Users\samy.timalsina.AUER\source\repos\VS2022AndLaterRepos\repos\Python\AnomalyApi
uvicorn anomalyApp:anomalyApp --host 0.0.0.0 --port 105 --workers 1