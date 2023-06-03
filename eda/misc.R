# ####################################################################################
# Misc

# dt_by_col <- cdc_survey_data_trim[, .(N = length(unique(SEQN))), by = column]
# setnames(dt_by_col, c('group_by_column', 'N'))
# dt_by_col <- dt_by_col[!is.na(group_by_column)]
# dt_by_col[, variable := column]
# dt_by_col[, group_by_column := round(group_by_column)]
# dcast(dt_by_col, formula = variable ~ group_by_column, value.var = "N")
# 
# 
# 
# create_univariate_plot <- function(dt = cdc_survey_data_trim,
#                                    column,
#                                    label_name) {
#   # group by col
#   dt_by_col <-
#     cdc_survey_data_trim[, .(N = length(unique(SEQN))), by = column]
#   setnames(dt_by_col, c('group_by_column', 'N'))
#   dt_by_col <- dt_by_col[!is.na(group_by_column)]
#   dt_by_col[, pct_of_tot := N / sum(N)]
#   
#   if (column == "has_depression_medicine") {
#     label_name <- column
#   } else {
#     # for specific cols
#     preferred_levels <- c('Not at all',
#                           'Several days',
#                           'Half of the days',
#                           'Nearly every day')
#     dt_by_col$group_by_column <-
#       factor(dt_by_col$group_by_column,
#              levels = preferred_levels)
#     label_name <- sub('_desc', '', column)
#   }
#   
#   # plot
#   by_col_plot <- ggplot(data = dt_by_col,
#                         aes(x = group_by_column,
#                             y = pct_of_tot)) +
#     geom_bar(stat = 'identity') +
#     xlab(glue(label_name)) +
#     ylab("Pct of Responses") +
#     ggtitle(glue(label_name)) +
#     #scale_y_continuous(labels = scales::comma_format()) +
#     scale_y_continuous(labels = percent_format(accuracy = 5L), limits = c(0, 1)) +
#     theme(
#       axis.text = element_text(size = 8),
#       axis.title = element_text(size = 8),
#       plot.title = element_text(size = 10, face = 'bold')
#     )
#   
#   return(by_col_plot)
# }
# 
# create_univariate_plot_num <- function(dt = cdc_survey_data_trim,
#                                        column,
#                                        label_name) {
#   # group by col
#   dt_by_col <-
#     cdc_survey_data_trim[, .(N = length(unique(SEQN))), by = column]
#   setnames(dt_by_col, c('group_by_column', 'N'))
#   
#   if (column == "has_depression_medicine") {
#     label_name <- column
#   } else {
#     # for specific cols
#     preferred_levels <- c('Not at all',
#                           'Several days',
#                           'Half of the days',
#                           'Nearly every day')
#     dt_by_col$group_by_column <-
#       factor(dt_by_col$group_by_column,
#              levels = preferred_levels)
#     label_name <- sub('_desc', '', column)
#   }
#   
#   # plot
#   by_col_plot <- ggplot(data = dt_by_col,
#                         aes(x = group_by_column,
#                             y = N)) +
#     geom_bar(stat = 'identity') +
#     xlab(glue(label_name)) +
#     ylab("# of Responses") +
#     ggtitle(glue(label_name)) +
#     scale_y_continuous(labels = scales::comma_format()) +
#     #scale_y_continuous(labels = percent_format(accuracy = 5L), limits = c(0, 1)) +
#     theme(
#       axis.text = element_text(size = 8),
#       axis.title = element_text(size = 8),
#       plot.title = element_text(size = 10, face = 'bold')
#     )
#   
#   return(by_col_plot)
# }
# 
# 
# plt1 <- create_univariate_plot(
#   column = 'little_interest_in_doing_things'
# )
# 
# plt2 <- create_univariate_plot(
#   column = 'feeling_down_depressed_hopeless_desc'
# )
# 
# plt3 <- create_univariate_plot(
#   column = 'trouble_falling_or_staying_asleep_desc'
# )
# 
# plt4 <- create_univariate_plot(
#   column = 'feeling_tired_or_having_little_energy_desc'
# )
# 
# plt5 <- create_univariate_plot(
#   column = 'poor_appetitie_or_overeating_desc'
# )
# 
# plt6 <- create_univariate_plot(
#   column = 'feeling_bad_about_yourself_desc'
# )
# 
# plt7 <- create_univariate_plot(
#   column = 'trouble_concentrating_desc'
# )
# 
# plt8 <- create_univariate_plot(
#   column = 'moving_or_speaking_to_slowly_or_fast_desc'
# )
# 
# plt9 <- create_univariate_plot(
#   column = 'thoughts_you_would_be_better_off_dead_desc'
# )
# 
# plt10 <- create_univariate_plot(
#   column = 'difficult_doing_daytoday_tasks'
# )
# 
# plt11 <- create_univariate_plot(
#   column = 'has_depression_medicine'
# )
# 
# (plt1 + plt2) /
#   (plt3 + plt4)
# 
# (plt5 + plt6) /
#   (plt7 + plt8)
# 
# (plt9 + plt11)


# create_desc_column_difficulty <- function(x) {
#   case_when(
#     round(x) == 0 ~ "Not at all difficult",
#     round(x) == 1 ~ "Somewhat difficult",
#     round(x) == 2 ~ "Very difficult",
#     round(x) == 3 ~ "Extremely difficult"
#   )
# }
