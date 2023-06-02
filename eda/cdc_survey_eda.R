library(data.table)
library(ggplot2)
library(glue)
library(scales)
library(patchwork)
library(dplyr)

# ####################################################################################
# Load data


cdc_survey_data <- as.data.table(read.csv("pipeline/df_cdc_joined_clean.csv"))
cdc_data_mapping <- as.data.table(read.csv("pipeline/df_var_mapping.csv"))

# ####################################################################################
# Setup
