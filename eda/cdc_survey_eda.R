library(data.table)
library(ggplot2)
library(glue)
library(scales)
library(patchwork)
library(dplyr)

# ####################################################################################
# Load data

cdc_survey_data <- as.data.table(read.csv("data/cdc_nhanes_survey_responses_clean.csv"))

# ####################################################################################
# Candidate Target Variables

create_summary_table <- function(column) {
  dt_by_col <-
    cdc_survey_data[, .(N = length(unique(SEQN))), by = column]
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
  'MDD'
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
  'MDD',
  'alcoholic_drinks_past_12mo',
  'how_healthy_is_your_diet',
  "has_diabetes",                         
  "count_days_seen_doctor_12mo",                 
  "has_overweight_diagnosis",                    
  "count_days_moderate_recreational_activity",   
  "count_minutes_moderate_recreational_activity",
  "count_minutes_moderate_sedentary_activity",   
  "count_days_physical_activity_youth",             
  "has_tried_to_lose_weight_12mo",               
# "has_ate_less_to_lose_weight",                 
#  "has_exercised_to_lose_weight",                
  "count_lost_10plus_pounds",                      
  "count_tried_to_lose_weight_youth",            
  "has_been_pregnant",                           
  "food_security_level_household",               
  "food_security_level_adult",                   
  "has_health_insurance",                       
  "has_health_insurance_gap",                    
  "general_health_condition",                    
  "duration_last_healthcare_visit",              
  "monthly_poverty_index",                       
  "monthly_poverty_index_category",                 
  "count_hours_worked_last_week",                
  "has_smoked_tabacco_last_5days",              
  "is_male",                                      
  "age_in_years",                               
  "education_level",                             
  "is_usa_born"           
)

cor_matrix <-
  cor(cdc_survey_data[, mget(correlation_variables)], use = "pairwise.complete.obs")

fwrite(cor_matrix,"correlation_matrix.csv")



