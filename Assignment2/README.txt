All the analysis and training of model were done on Kaggle

To test the model, perform the following steps:
	1. Open nlpassignment2.ipynb
	2. If you are testing on Kaggle, add the dataset to the notebook from the link given in problem statement. Else replace the location of input csv from '/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv' to wherever you have stored the input csv file
	3. Download the trained model from the link provided in the pdf report
	4. Run all the cells present before the cell containing training loop. On the top of the training loop, you will be find a label saying "Model training starts here (Do not run this cell if you just want to load model and test)"
	5. Ignore the next cell which contains code to plot loss and score with epochs
	6. Next cell contains code to load the trained model. So replace the location '/kaggle/working/model.pt' to wherever you have stored the trained model
	7. Run the cells present after that which will load the model and run it on the test set. You will get the F1 score of the model on the test set.
	
If you want to re-train the model, just comment out the cell containing code to load the trained model and run all the cells. Training won't take more than 5 minutes.
