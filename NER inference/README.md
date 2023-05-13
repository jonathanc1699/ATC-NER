# NER Inference

Steps to reproduce NER inference:
1. Install virtual environment tool virtualenv using python3.8 -m pip install virtualenv
2. Create a python 3 virtual environment using virtualenv -p python3.8 venv
3. Activate the virtual environment using source venv/bin/activate
4. Install requirements - other than previous one - fastapi; uvicorn (https://fastapi.tiangolo.com/tutorial/)
5. Run the app using : uvicorn inference_script:app --reload ( check that the directories in the script are correct and you are in the main folder )
6. Test the API endpoints using swagger (http://127.0.0.1:8000/docs); POSTman can also be used for testing.

Note: when using with Async system, ensure that the NER app is running before starting speech-to-text transcription