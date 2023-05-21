library(data.table)
library(ggplot2)
library(glue)
library(scales)
library(patchwork)
library(dplyr)

# ####################################################################################
# Load data


ppd_survey_data <- as.data.table(read.csv("data/kaggle_prams_ppd_surveydata.csv"))


# ####################################################################################
# Setup

create_univariate_plot <- function(dt = ppd_survey_data,
                                   column) {
  dt_by_col <- dt[, .N, by = c(column)][order(mget(column))]
  dt_by_col[, pct_of_tot := N / sum(N)]
  setnames(dt_by_col, c('group_by_column', 'N', 'pct_of_tot'))
  
  by_col_plot <- ggplot(data = dt_by_col,
                        aes(x = group_by_column,
                            y = pct_of_tot)) +
    geom_bar(stat = 'identity') +
    xlab(glue(column)) +
    ylab("Pct of Responses") +
    ggtitle(glue("{column}")) +
    scale_y_continuous(labels = percent_format(accuracy = 5L), limits = c(0,1))+
    theme(axis.text=element_text(size=8),
          axis.title=element_text(size=8),
          plot.title = element_text(size=10, face='bold'))
  
  return(by_col_plot)
}

setnames(
  ppd_survey_data,
  new = c(
    'Timestamp',
    'Age',
    'Feeling Sad/Tearful',
    'Irratable Towards Baby/Partner',
    'Touble Sleeping',
    'Problems Concentrating/Decision Making',
    'Overeating or Loss of Appetite',
    'Feeling Anxious',
    'Feeling of Guilt',
    'Problems Bonding With Baby',
    'Has Suicide Attempt'
  )
)
columns <- colnames(ppd_survey_data)

# create numeric column for suicide attempt
ppd_survey_data[, has_suicide_attempt := ifelse(
  `Has Suicide Attempt` == "Yes", 1,
  ifelse(`Has Suicide Attempt` == "No", 0, NA)
)]

global_avg <- mean(ppd_survey_data$has_suicide_attempt,na.rm=TRUE)


create_univariate_plot_pct_target <- function(dt = ppd_survey_data,
                                              column) {
  dt_by_col <-
    dt[, .(pct_suicide_attemmpt = mean(has_suicide_attempt, na.rm = TRUE)),
       by = c(column)][order(mget(column))]
  setnames(dt_by_col, c('group_by_column', 'pct_suicide_attemmpt'))
  
  by_col_plot <- ggplot(data = dt_by_col,
                        aes(x = group_by_column,
                            y = pct_suicide_attemmpt)) +
    geom_bar(stat = 'identity') +
    xlab(glue(column)) +
    ylab("Pct Attempted Suicide") +
    ggtitle(glue("{column}")) +
    scale_y_continuous(labels = percent_format(accuracy = 5L), limits = c(0,1)) +
    theme(
      axis.text = element_text(size = 8),
      axis.title = element_text(size = 8),
      plot.title = element_text(size = 10, face = 'bold')
    ) +
    geom_hline(yintercept = global_avg, col = 'grey')
  
  return(by_col_plot)
}


# ####################################################################################
# MISC EDA

str(ppd_survey_data)

head(ppd_survey_data)

# ####################################################################################
# Univariate Plots (Sample Size)

by_col1 <- create_univariate_plot(
  column = columns[2]
)

by_col2 <- create_univariate_plot(
  column = columns[3]
)

by_col3 <- create_univariate_plot(
  column = columns[4]
)

by_col4 <- create_univariate_plot(
  column = columns[5]
)

by_col5 <- create_univariate_plot(
  column = columns[6]
)

by_col6 <- create_univariate_plot(
  column = columns[7]
)

by_col7 <- create_univariate_plot(
  column = columns[8]
)

by_col8 <- create_univariate_plot(
  column = columns[9]
)

by_col9 <- create_univariate_plot(
  column = columns[10]
)

by_col10 <- create_univariate_plot(
  column = columns[11]
)

(by_col1 + by_col2 + by_col3) /
  (by_col4 + by_col5 + by_col6) /
  (by_col7 + by_col8 + by_col9)

by_col10



# ####################################################################################
# Univariate Plots (% Suicide Attempt)

by_col1 <- create_univariate_plot_pct_target(
  column = columns[2]
)

by_col2 <- create_univariate_plot_pct_target(
  column = columns[3]
)

by_col3 <- create_univariate_plot_pct_target(
  column = columns[4]
)

by_col4 <- create_univariate_plot_pct_target(
  column = columns[5]
)

by_col5 <- create_univariate_plot_pct_target(
  column = columns[6]
)

by_col6 <- create_univariate_plot_pct_target(
  column = columns[7]
)

by_col7 <- create_univariate_plot_pct_target(
  column = columns[8]
)

by_col8 <- create_univariate_plot_pct_target(
  column = columns[9]
)

by_col9 <- create_univariate_plot_pct_target(
  column = columns[10]
)

by_col10 <- create_univariate_plot_pct_target(
  column = columns[11]
)

(by_col1 + by_col2 + by_col3) /
  (by_col4 + by_col5 + by_col6) /
  (by_col7 + by_col8 + by_col9)
