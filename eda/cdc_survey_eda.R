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
# Clean up Columns

#cdc_survey_cols <- colnames(cdc_survey_data)
#depression_cols <- cdc_survey_cols[grepl("DPQ",cdc_survey_cols)]
#prescription_cols <- cdc_survey_cols[grepl("RXDR",cdc_survey_cols)]
#cdc_data_mapping[Variable.Name %in% depression_cols][1:10,]

column_mapping <- data.table(
  old = c(
    "DPQ010",
    "DPQ020",
    "DPQ030",
    "DPQ040",
    "DPQ050",
    "DPQ060",
    "DPQ070",
    "DPQ080",
    "DPQ090",
    "DPQ100",
    "RXDRSC1",
    "RXDRSC2",
    "RXDRSC3",
    "RXDRSD1",
    "RXDRSD2",
    "RXDRSD3",
    "ALQ121",
    "DBQ700",
    "DIQ010",
    "DID250",
    "MCQ080",
    "PAQ670",
    "PAD675",
    "PAD680",
    "PAQ706",
    "WHQ030",
    "WHQ070",
    "WHD080A",
    "WHD080D",
    "WHQ225",
    "WHQ030M",
    "WHQ520",
    "RHQ131",
    "FSDHH",
    "FSDAD",
    "HIQ011",
    "HIQ210",
    "HUQ010",
    "HUD062",
    "INDFMMPI",
    "INDFMMPC",
 #   "HOQ065",
    "OCQ180",
    "SMQ681",
    "RIAGENDR",
    "RIDAGEYR",
    "DMDEDUC2",
    "DMDBORN4"
  ),
  new = c(
    'little_interest_in_doing_things',
    'feeling_down_depressed_hopeless',
    'trouble_falling_or_staying_asleep',
    'feeling_tired_or_having_little_energy',
    'poor_appetitie_or_overeating',
    'feeling_bad_about_yourself',
    'trouble_concentrating',
    'moving_or_speaking_to_slowly_or_fast',
    'thoughts_you_would_be_better_off_dead',
    'difficult_doing_daytoday_tasks',
    'ICD_10_CM_code1',
    'ICD_10_CM_code2',
    'ICD_10_CM_code3',
    'ICD_10_CM_code1_desc',
    'ICD_10_CM_code2_desc',
    'ICD_10_CM_code3_desc',
    "alcoholic_drinks_past_12mo",
    "how_healthy_is_your_diet",
    "has_diabetes",
    "count_days_seen_doctor_12mo",
    "has_overweight_diagnosis",
    "count_days_moderate_recreational_activity",
    "count_minutes_moderate_recreational_activity",
    "count_minutes_moderate_sedentary_activity",
    "count_days_physical_activity_youth",
    "how_do_you_consider_your_weight",
    "has_tried_to_lose_weight_12mo",
    "has_ate_less_to_lose_weight",
    "has_exercised_to_lose_weight",
    "count_lost_10plus_pounds",
    "how_do_you_consider_your_weight_youth",
    "count_tried_to_lose_weight_youth",
    "has_been_pregnant",
    "food_security_level_household",
    "food_security_level_adult",
    "has_health_insurance",
    "has_health_insurance_gap",
    "general_health_condition",
    "duration_last_healthcare_visit",
    "monthly_poverty_index",
    "monthly_poverty_index_level",
 #   "housing_characteristics",
    "count_hours_worked_last_week",
    "has_smoked_tabacco_last_5days",
    "gender",
    "age_in_years",
    "education_level",
    "is_usa_born"
  )
)

columns_to_keep <- c(
  'SEQN',
  column_mapping$old
)

cdc_survey_data_trim <- cdc_survey_data[,mget(columns_to_keep)]

# ####################################################################################
# sanitize data

setnames(
  cdc_survey_data_trim,
  old = column_mapping$old,
  new = column_mapping$new
)

# ##############################
# candidate depression variables


# use prescription columns
cdc_survey_data_trim[, has_depression_medicine := case_when(
  grepl("F32|F33",ICD_10_CM_code1) ~ 1,
  grepl("F32|F33",ICD_10_CM_code2) ~ 1,
  grepl("F32|F33",ICD_10_CM_code3) ~ 1,
  !is.na(ICD_10_CM_code1) ~ 0,
  !is.na(ICD_10_CM_code2) ~ 0,
  !is.na(ICD_10_CM_code3) ~ 0
)]

# remove values that represent "don't know" or "refused"
# for most the legend is
# 0 = Not at All
# 1 = Several Days
# 2 = Half of the Days
# 3 = Nearly Every Day
# for difficulty question is slightly different but similar

create_clean_column <- function(x) {
  case_when(
    round(x) == 0 ~ 0,
    round(x) == 1 ~ 1,
    round(x) == 2 ~ 2,
    round(x) == 3 ~ 3
  )
}

cdc_survey_data_trim[, little_interest_in_doing_things := create_clean_column(little_interest_in_doing_things)]
cdc_survey_data_trim[, feeling_down_depressed_hopeless := create_clean_column(feeling_down_depressed_hopeless)]
cdc_survey_data_trim[, trouble_falling_or_staying_asleep := create_clean_column(trouble_falling_or_staying_asleep)]
cdc_survey_data_trim[, feeling_tired_or_having_little_energy := create_clean_column(feeling_tired_or_having_little_energy)]
cdc_survey_data_trim[, poor_appetitie_or_overeating := create_clean_column(poor_appetitie_or_overeating)]
cdc_survey_data_trim[, feeling_bad_about_yourself := create_clean_column(feeling_bad_about_yourself)]
cdc_survey_data_trim[, trouble_concentrating := create_clean_column(trouble_concentrating)]
cdc_survey_data_trim[, moving_or_speaking_to_slowly_or_fast := create_clean_column(moving_or_speaking_to_slowly_or_fast)]
cdc_survey_data_trim[, thoughts_you_would_be_better_off_dead := create_clean_column(thoughts_you_would_be_better_off_dead)]
cdc_survey_data_trim[, difficult_doing_daytoday_tasks := create_clean_column(difficult_doing_daytoday_tasks)]

# ##############################
# candidate risk factors

cdc_survey_data_trim[, alcoholic_drinks_past_12mo := ifelse(alcoholic_drinks_past_12mo <= 10,
                                                            round(alcoholic_drinks_past_12mo),
                                                            NA_integer_)]
cdc_survey_data_trim[, how_healthy_is_your_diet := ifelse(how_healthy_is_your_diet <= 5,
                                                          round(how_healthy_is_your_diet),
                                                          NA_integer_)]

cdc_survey_data_trim[, has_diabetes := case_when(has_diabetes == 1 ~ 1,
                                                 has_diabetes == 2 ~ 0)]

cdc_survey_data_trim[, count_days_seen_doctor_12mo := ifelse(count_days_seen_doctor_12mo <= 40,
                                                             round(count_days_seen_doctor_12mo),
                                                             NA_integer_)]

cdc_survey_data_trim[, has_overweight_diagnosis := case_when(has_overweight_diagnosis == 1 ~ 1,
                                                             has_overweight_diagnosis == 2 ~ 0)]


cdc_survey_data_trim[, count_days_moderate_recreational_activity := ifelse(count_days_moderate_recreational_activity <= 7,
                                                             round(count_days_moderate_recreational_activity),
                                                             NA_integer_)]

cdc_survey_data_trim[, count_minutes_moderate_recreational_activity := ifelse(count_minutes_moderate_recreational_activity <= 600,
                                                                           round(count_minutes_moderate_recreational_activity),
                                                                           NA_integer_)]

cdc_survey_data_trim[, count_minutes_moderate_sedentary_activity := ifelse(count_minutes_moderate_sedentary_activity <= 1320,
                                                                              round(count_minutes_moderate_sedentary_activity),
                                                                              NA_integer_)]

cdc_survey_data_trim[, count_days_physical_activity_youth := ifelse(count_days_physical_activity_youth <= 7,
                                                                           round(count_days_physical_activity_youth),
                                                                           NA_integer_)]


# cdc_survey_data_trim[, how_do_you_consider_your_weight := case_when(how_do_you_consider_your_weight == 1 ~ "Overweight",
#                                                                     how_do_you_consider_your_weight == 2 ~ "Underweight",
#                                                                     how_do_you_consider_your_weight == 3 ~ "About right weight")]

cdc_survey_data_trim[, has_tried_to_lose_weight_12mo := case_when(has_tried_to_lose_weight_12mo == 1 ~ 1,
                                                                  has_tried_to_lose_weight_12mo == 2 ~ 0)]

cdc_survey_data_trim[, has_ate_less_to_lose_weight := case_when(has_tried_to_lose_weight_12mo == 1 & has_ate_less_to_lose_weight == 10 ~ 1,
                                                                has_tried_to_lose_weight_12mo == 1 ~ 0)]

cdc_survey_data_trim[, has_exercised_to_lose_weight := case_when(has_tried_to_lose_weight_12mo == 1 & has_exercised_to_lose_weight == 12 ~ 1,
                                                                has_tried_to_lose_weight_12mo == 1 ~ 0)]

cdc_survey_data_trim[, count_lost_10plus_pounds := ifelse(count_lost_10plus_pounds <= 4,
                                                                    round(count_lost_10plus_pounds),
                                                                    NA_integer_)]

# cdc_survey_data_trim[, how_do_you_consider_your_weight_youth := case_when(how_do_you_consider_your_weight_youth == 1 ~ "Overweight",
#                                                                           how_do_you_consider_your_weight_youth == 2 ~ "Underweight",
#                                                                           how_do_you_consider_your_weight_youth == 3 ~ "About right weight")]

cdc_survey_data_trim[, count_tried_to_lose_weight_youth := ifelse(count_tried_to_lose_weight_youth <= 3,
                                                          round(count_tried_to_lose_weight_youth),
                                                          NA_integer_)]

cdc_survey_data_trim[, has_been_pregnant := case_when(has_been_pregnant == 1 ~ 1,
                                                      has_been_pregnant == 2 ~ 0)]

cdc_survey_data_trim[, food_security_level := case_when(has_been_pregnant == 1 ~ 1,
                                                      has_been_pregnant == 2 ~ 0)]

cdc_survey_data_trim[, food_security_level_household := ifelse(food_security_level_household <= 4,
                                                         round(food_security_level_household),
                                                         NA_integer_)]

cdc_survey_data_trim[, food_security_level_adult := ifelse(food_security_level_adult <= 4,
                                                               round(food_security_level_adult),
                                                               NA_integer_)]

cdc_survey_data_trim[, has_health_insurance := case_when(has_health_insurance == 1 ~ 1,
                                                         has_health_insurance == 2 ~ 0)]

cdc_survey_data_trim[, has_health_insurance_gap := case_when(has_health_insurance_gap == 1 ~ 1,
                                                             has_health_insurance_gap == 2 ~ 0)]


cdc_survey_data_trim[, general_health_condition := ifelse(general_health_condition <= 5,
                                                           round(general_health_condition),
                                                           NA_integer_)]

cdc_survey_data_trim[, duration_last_healthcare_visit := ifelse(duration_last_healthcare_visit <= 4,
                                                          round(duration_last_healthcare_visit),
                                                          NA_integer_)]

cdc_survey_data_trim[, duration_last_healthcare_visit := ifelse(duration_last_healthcare_visit <= 4,
                                                                round(duration_last_healthcare_visit),
                                                                NA_integer_)]

cdc_survey_data_trim[, monthly_poverty_index_level := ifelse(monthly_poverty_index_level <= 3,
                                                                round(monthly_poverty_index_level),
                                                                NA_integer_)]

cdc_survey_data_trim[, count_hours_worked_last_week := ifelse(count_hours_worked_last_week <= 80,
                                                             round(count_hours_worked_last_week),
                                                             NA_integer_)]

cdc_survey_data_trim[, has_smoked_tabacco_last_5days := case_when(has_smoked_tabacco_last_5days == 1 ~ 1,
                                                                  has_smoked_tabacco_last_5days == 2 ~ 0)]

# 0 = male
cdc_survey_data_trim[, gender := case_when(gender == 1 ~ 0,
                                           gender == 2 ~ 1)]


cdc_survey_data_trim[, age_in_years := ifelse(age_in_years <= 80,
                                              round(age_in_years),
                                              NA_integer_)]

cdc_survey_data_trim[, education_level := ifelse(education_level <= 5,
                                                 round(education_level),
                                                 NA_integer_)]

cdc_survey_data_trim[, is_usa_born := case_when(is_usa_born == 1 ~ 1,
                                           is_usa_born == 2 ~ 0)]



# ####################################################################################
# Candidate Target Variables


create_summary_table <- function(column) {
  dt_by_col <-
    cdc_survey_data_trim[, .(N = length(unique(SEQN))), by = column]
  setnames(dt_by_col, c('group_by_column', 'N'))
  dt_by_col <- dt_by_col[!is.na(group_by_column)]
  dt_by_col[, variable := column]
  dt_by_col[, group_by_column := round(group_by_column)]
  dt_by_col <-
    dcast(dt_by_col,
          formula = variable ~ group_by_column,
          value.var = "N")
  return(dt_by_col)
}

columns <- c(    
  'little_interest_in_doing_things',
  'feeling_down_depressed_hopeless',
  'trouble_falling_or_staying_asleep',
  'feeling_tired_or_having_little_energy',
  'poor_appetitie_or_overeating',
  'feeling_bad_about_yourself',
  'trouble_concentrating',
  'moving_or_speaking_to_slowly_or_fast',
  'thoughts_you_would_be_better_off_dead',
  'difficult_doing_daytoday_tasks',
  'has_depression_medicine'
)

candidate_target_variable <- rbindlist(lapply(columns, function(x) {
  create_summary_table(x)
}), fill = TRUE)

fwrite(candidate_target_variable,"candidate_target_variable_stats.csv")

# ####################################################################################
# Correlation plot

correlation_variables <- c(    
  'little_interest_in_doing_things',
  'feeling_down_depressed_hopeless',
  'trouble_falling_or_staying_asleep',
  'feeling_tired_or_having_little_energy',
  'poor_appetitie_or_overeating',
  'feeling_bad_about_yourself',
  'trouble_concentrating',
  'moving_or_speaking_to_slowly_or_fast',
  'thoughts_you_would_be_better_off_dead',
  'difficult_doing_daytoday_tasks',
  'has_depression_medicine',
  'alcoholic_drinks_past_12mo',
  'how_healthy_is_your_diet',
  "has_diabetes",                         
  "count_days_seen_doctor_12mo",                 
  "has_overweight_diagnosis",                    
  "count_days_moderate_recreational_activity",   
  "count_minutes_moderate_recreational_activity",
  "count_minutes_moderate_sedentary_activity",   
  "count_days_physical_activity_youth",          
  "how_do_you_consider_your_weight",             
  "has_tried_to_lose_weight_12mo",               
  "has_ate_less_to_lose_weight",                 
  "has_exercised_to_lose_weight",                
  "count_lost_10plus_pounds",                    
  "how_do_you_consider_your_weight_youth",      
  "count_tried_to_lose_weight_youth",            
  "has_been_pregnant",                           
  "food_security_level_household",               
  "food_security_level_adult",                   
  "has_health_insurance",                       
  "has_health_insurance_gap",                    
  "general_health_condition",                    
  "duration_last_healthcare_visit",              
  "monthly_poverty_index",                       
  "monthly_poverty_index_level",                 
  "count_hours_worked_last_week",                
  "has_smoked_tabacco_last_5days",              
  "gender",                                      
  "age_in_years",                               
  "education_level",                             
  "is_usa_born",                                 
  "has_depression_medicine",                    
  "food_security_level" 
)

cor_matrix <-
  cor(cdc_survey_data_trim[, mget(correlation_variables)], use = "pairwise.complete.obs")

fwrite(cor_matrix,"correlation_matrix.csv")



