library(data.table)
library(ggplot2)
library(glue)
library(scales)
library(patchwork)
library(dplyr)

# ####################################################################################
# Load data


cdc_survey_data <- as.data.table(read.csv("pipeline/df_cdc_joined_clean_pre.csv"))

# ####################################################################################
# Data imputation

# ##############################
# candidate depression variables

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

cdc_survey_data[, little_interest_in_doing_things := create_clean_column(DPQ010)]
cdc_survey_data[, feeling_down_depressed_hopeless := create_clean_column(DPQ020)]
cdc_survey_data[, trouble_falling_or_staying_asleep := create_clean_column(DPQ030)]
cdc_survey_data[, feeling_tired_or_having_little_energy := create_clean_column(DPQ040)]
cdc_survey_data[, poor_appetitie_or_overeating := create_clean_column(DPQ050)]
cdc_survey_data[, feeling_bad_about_yourself := create_clean_column(DPQ060)]
cdc_survey_data[, trouble_concentrating := create_clean_column(DPQ070)]
cdc_survey_data[, moving_or_speaking_to_slowly_or_fast := create_clean_column(DPQ080)]
cdc_survey_data[, thoughts_you_would_be_better_off_dead := create_clean_column(DPQ090)]
cdc_survey_data[, difficult_doing_daytoday_tasks := create_clean_column(DPQ100)]

# ##############################
# candidate risk factors


cdc_survey_data[, alcoholic_drinks_past_12mo := ifelse(ALQ121 <= 10,
                                                       round(ALQ121),
                                                       NA_integer_)]
cdc_survey_data[, how_healthy_is_your_diet := ifelse(DBQ700 <= 5,
                                                     round(DBQ700),
                                                     NA_integer_)]

cdc_survey_data[, has_diabetes := case_when(DIQ010 == 1 ~ 1,
                                            DIQ010 == 2 ~ 0)]

cdc_survey_data[, count_days_seen_doctor_12mo := ifelse(DID250 <= 40,
                                                        round(DID250),
                                                        NA_integer_)]

cdc_survey_data[, has_overweight_diagnosis := case_when(MCQ080 == 1 ~ 1,
                                                        MCQ080 == 2 ~ 0)]


cdc_survey_data[, count_days_moderate_recreational_activity := ifelse(PAQ670 <= 7,
                                                                      round(PAQ670),
                                                                      NA_integer_)]

cdc_survey_data[, count_minutes_moderate_recreational_activity := ifelse(PAD675 <= 600,
                                                                         round(PAD675),
                                                                         NA_integer_)]

cdc_survey_data[, count_minutes_moderate_sedentary_activity := ifelse(PAD680 <= 1320,
                                                                      round(PAD680),
                                                                      NA_integer_)]

cdc_survey_data[, count_days_physical_activity_youth := ifelse(PAQ706 <= 7,
                                                               round(PAQ706),
                                                               NA_integer_)]


cdc_survey_data[, has_tried_to_lose_weight_12mo := case_when(WHQ070 == 1 ~ 1,
                                                             WHQ070 == 2 ~ 0)]

cdc_survey_data[, has_ate_less_to_lose_weight := case_when(WHQ070 == 1 &
                                                             WHD080A == 10 ~ 1,
                                                           WHD080A == 1 ~ 0)]

cdc_survey_data[, has_exercised_to_lose_weight := case_when(WHQ070 == 1 &
                                                              WHD080D == 12 ~ 1,
                                                            WHD080D == 1 ~ 0)]

cdc_survey_data[, count_lost_10plus_pounds := ifelse(WHQ225 <= 4,
                                                     round(WHQ225),
                                                     NA_integer_)]

cdc_survey_data[, count_tried_to_lose_weight_youth := ifelse(WHQ520 <= 3,
                                                             round(WHQ520),
                                                             NA_integer_)]

cdc_survey_data[, has_been_pregnant := case_when(RHQ131 == 1 ~ 1,
                                                 RHQ131 == 2 ~ 0)]


cdc_survey_data[, food_security_level_household := ifelse(FSDHH <= 4,
                                                          round(FSDHH),
                                                          NA_integer_)]

cdc_survey_data[, food_security_level_adult := ifelse(FSDAD <= 4,
                                                      round(FSDAD),
                                                      NA_integer_)]

cdc_survey_data[, has_health_insurance := case_when(HIQ011 == 1 ~ 1,
                                                    HIQ011 == 2 ~ 0)]

cdc_survey_data[, has_health_insurance_gap := case_when(HIQ210 == 1 ~ 1,
                                                        HIQ210 == 2 ~ 0)]


cdc_survey_data[, general_health_condition := ifelse(HUQ010 <= 5,
                                                     round(HUQ010),
                                                     NA_integer_)]

cdc_survey_data[, duration_last_healthcare_visit := ifelse(HUD062 <= 4,
                                                           round(HUD062),
                                                           NA_integer_)]

cdc_survey_data[, monthly_poverty_index := INDFMMPI]


cdc_survey_data[, monthly_poverty_index_category := ifelse(INDFMMPC <= 3,
                                                        round(INDFMMPC),
                                                        NA_integer_)]

cdc_survey_data[, count_hours_worked_last_week := ifelse(OCQ180 <= 80,
                                                         round(OCQ180),
                                                         NA_integer_)]

cdc_survey_data[, has_smoked_tabacco_last_5days := case_when(SMQ681 == 1 ~ 1,
                                                             SMQ681 == 2 ~ 0)]

# 0 = male
cdc_survey_data[, gender := case_when(RIAGENDR == 1 ~ 0,
                                      RIAGENDR == 2 ~ 1)]


cdc_survey_data[, age_in_years := ifelse(RIDAGEYR <= 80,
                                         round(RIDAGEYR),
                                         NA_integer_)]

cdc_survey_data[, education_level := ifelse(DMDEDUC2 <= 5,
                                            round(DMDEDUC2),
                                            NA_integer_)]

cdc_survey_data[, is_usa_born := case_when(DMDBORN4 == 1 ~ 1,
                                           DMDBORN4 == 2 ~ 0)]





fwrite(cdc_survey_data,"./pipeline/df_cdc_clean.csv")