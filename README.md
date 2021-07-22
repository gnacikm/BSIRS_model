This code can be used to generate the figures from the paper **Implied infection-cutting behaviour from a spatial game**
available on arxiv [ link](https://arxiv.org/). 

All the code is located in the *code* folder. 

# Model - BSIRS
The model (section 2 of the paper) is contained in *bsirs_model.py*, the exposure matrix is contained in *exposure_matrix.py.* 
To generate all the outputs (calibrated parameters) of the model, use *simulation_results.ipynb*. This uses *execute_system.py* to evolve the system.
The obtained varables (average infection-cutting rate u, infection number etc.) are saved inside *static>saved_params*.

# Figures
The figures that are available in the paper are generated inside Jupyter Notebooks (notebooks folder) then they are saved into *static>saved_figures folder*.

# Data (csv and shp)
**The NHS Covid-19 data** and all other data used in this work is contained in the folder *static>utla_data*. These are either csv files or shape files (shp). In particular, the processed NHS data is contained in the file *static/utla_data/shp/covid-cases_EN_recent.shp*; these contains the number of active Covid-19 cases in each UTLA in England between 01-07-2020 - 02-16-2021. This was obtained by processing the UTLA data of Covid-19 cases from [https://coronavirus.data.gov.uk/](https://coronavirus.data.gov.uk/).

To generate active cases directly from [https://coronavirus.data.gov.uk/](https://coronavirus.data.gov.uk/)  please see the notebook *generate_active_cases.ipynb*.
