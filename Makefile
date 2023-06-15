# Clone the original Obermeyer et al. repo and patch it to
# add code to extract a few versions of the public dataset
# and compute metrics for an index model
dissecting-bias:
	git clone https://gitlab.com/labsysmed/dissecting-bias.git
	cd dissecting-bias && git apply ../update_to_dissecting_bias_repo.patch
	cd dissecting-bias/code/model && python3 extract_csvs_and_run_ols.py
	@echo '########################################'
	@echo
	@echo Dissecting Bias repo cloned and updated!
	@echo Datasets are in data/
	@echo
	@echo '########################################'

# Copy the full Obermeyer data from the dissecting-bias repo into data/
data/dissecting_bias_dataset_all_outcomes_and_features.csv: dissecting-bias
	cd data && cp ../dissecting-bias/data/dissecting_bias_dataset_all_outcomes_and_features.csv .

# Copy an ols-friendly version of the data from the dissecting-bias repo into data/
data/dissecting_bias_dataset_all_outcomes_ols_friendly_features.csv: dissecting-bias
	cd data && cp ../dissecting-bias/data/dissecting_bias_dataset_all_outcomes_ols_friendly_features.csv .

# Copy a smaller subset of features from the dissecting-bias repo into data/
data/dissecting_bias_dataset_three_outcomes_limited_features_train_and_holdout.csv: dissecting-bias
	cd data && cp ../dissecting-bias/data/dissecting_bias_dataset_three_outcomes_limited_features_train_and_holdout.csv .

##### Figure 1 #####

# Figure 1: Run the fairness-maximizing MIP to find the best index model on the Obermeyer et al. dataset
dissecting-bias/results/maximize_index_model_results.csv: train_fairness_multitarget_mip.py figure1-settings.json data/dissecting_bias_dataset_all_outcomes_ols_friendly_features.csv
	python3 train_fairness_multitarget_mip.py figure1-settings.json
	cd dissecting-bias/results && cp ../../results/maximize_index_results_tuneK3_tuneN30_dissecting_bias_dataset_all_outcomes_ols_friendly_features.csv maximize_index_model_results.csv

# Figure 1: Compute and plot the results of the fairness-maximizing MIP against the original Obermeyer et al. results
figure1: dissecting-bias/results/maximize_index_model_results.csv dissecting-bias/code/table2_index_model.py plot_figure1.R
	cd dissecting-bias/code && python3 table2_index_model.py
	cd results && cp ../dissecting-bias/results/dissecting_bias_index_model_concentration_metric.csv .
	Rscript plot_figure1.R

##### Figure 2 #####

# Figure 2: Create semi-synthetic versions of the Obermeyer data
data/dissecting_bias_dataset_semisynthetic%: data/dissecting_bias_dataset_three_outcomes_limited_features_train_and_holdout.csv generate_semisynthetic_data.R
	Rscript generate_semisynthetic_data.R

# Figure 2: Run the fairness-maximizing MIP on each semi-synthetic version of the Obermeyer data
results/dissecting_bias_dataset_semisynthetic%: results/dissecting_bias_dataset_semisynthetic% train_fairness_multitarget_mip.py figure2-settings.json
	for b in `seq -f %.1f -1.0 0.1 1.0`; do \
		python3 train_fairness_multitarget_semisynthetic.py figure2-settings.json $$b; \
	done
	cd results && cat dissecting_bias_dataset_semisynthetic_*alpha*tune*.csv | sort | uniq > dissecting_bias_dataset_semisynthetic_tuneK30_tuneN300_allbvalues.csv

# Figure 2: Plot the results of the fairness-maximizing MIP on each semi-synthetic version of the Obermeyer data
figure2: results/dissecting_bias_dataset_semisynthetic% plot_figure2.R
	Rscript plot_figure2.R

clean:
	rm -rf dissecting-bias
	rm data/dissecting_bias_dataset_*outcomes*feature*.csv
	rm figures/healthcare_concentration_table_as_bar_plot.{png,pdf}
