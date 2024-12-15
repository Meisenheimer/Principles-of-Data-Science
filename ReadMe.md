**If there is any problem, please email me at zeyu-asparagine.wang@connect.polyu.hk .**

Requirement:

- ```C++```: The codes for preprocessing are written in C++ and the only dependency (the Eigen) has been included in ```./Preprocessing/```.

- ```Python```: All the packages needed are commonly used (e.g. PyTorch, scikit-learn, matplotlib, ...)

---

The followings are description for each folder (there are bash scripts for each code, you can check it before running):

- ```./Data/```: ```list.csv``` and ```subjectIDs.txt``` include some basic informations for the dataset, and ```./node_timeseries/``` contains the time series data given by HCP dataset. The ```./functional_connectivity/``` contains the functional connectivity which is computed by the program in ```./Preprocessing/```.

- ```./Preprocessing/```: Codes for preprocessing in C++, and depend on Eigen, which is included in the directory. The compile command is ```g++ main.cpp -o main.exe -O2 -Wall -Wextra -fopenmp``` and running by ```./main [node_number] [size] [step] [rho]```, where size and step are parameters for sliding window and rho is the parameter for ridge regression (where the algorithm is the same as the one given by HCP dataset). The program will load the time series data from ```./Data/``` and also save the results to it. **This should be run first (or just use the data given in ```./Data/functional_connectivity/```).**

- ```./CNN/```: Codes for CNN model, where the ```make.py``` is used for generate the bash scripts and the ```train.py``` is the program entry point.

- ```./Result/```: Includes the results of CNN model running on our machines. The accurancy and AUC for both train set and test set is recorded in. And the ```draw.py``` file will read the ```.txt``` files for visualization.

- ```./SVM```: Codes for SVM model, all are similar to ```./CNN/```, but the results are included in.

- ```./Analysis/``` and ```./Analysis-unused/```: Code for the Analysis part, including the codes for visualize the dynamic connectivity, the LDA and the most significant brain region. Codes in ```./Analysis-unused/``` are some analysis we did but not used in the report, including the PCA and normality test.

- ```./Slide/```: The slide for the presentation by LaTeX.

- ```./Report/```: The final report by LaTeX.

There are some ```draw.py``` with the results file, which is used to visualize the results saved in .txt file.