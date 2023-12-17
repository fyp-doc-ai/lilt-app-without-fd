1. Create a virtualenv
	`virtualenv lilt-env`
2. Install packages
	`pip install -r requirements.txt`
3. Dowload the models by running `dowload_model.ipynb` notebook
3. Run the app
	`uvicorn main:app --reload`