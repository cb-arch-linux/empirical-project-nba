# Automates the full project pipeline from data collection through to report
# rendering. Each target corresponds to one stage of the pipeline. Run "make all"
# to execute every stage in the correct order.

# make all          - Run the complete pipeline
# make collect      - Collect raw data from NBA API and CraftedNBA
# make clean_data   - Clean and merge raw data files
# make analysis     - Run regression model and generate MVP scores
# make figures      - Generate all plots and tables

# Create and activate the conda environment before running any targets

PYTHON = python

.PHONY: all collect clean_data analysis figures report

all: collect clean_data analysis figures report

collect:
	$(PYTHON) src/00_collect.py

clean_data:
	$(PYTHON) src/01_clean.py

analysis:
	$(PYTHON) src/02_analysis.py

figures:
	$(PYTHON) src/03_figures.py
