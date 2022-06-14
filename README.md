# EthovisionPathAnalysis_HDF5
Processes raw data from Ethovision, stores and analyses data in HDF5 format (actually, using mat files for now)

Python = 3.8
Required modules in `environment.yml`

To use this code, first export and save your raw data from Ethovision as excel files
Put excel files into the folder `data/rawData`
Update `documentation.cvs` with experiment info
update `lib_experiment_parameters.py` in the `modules` folder with experiment info

- `process_data_to_mat.py`	run this file to convert excel data to mat files
- `plot_mouse_trajectory.py`	plots a single mouse trajectory based on experiment, mouse number, and trial name
- `plot_LearningCurve.py`	plots the latency, distance, or speed of mouse trials over a single experiment