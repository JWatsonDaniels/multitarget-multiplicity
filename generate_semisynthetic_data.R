library(tidyverse)
library(modelr)
library(scales)

theme_set(theme_bw())

################################################################################
# Define functions
################################################################################

# a function to fit a parabola y ~ ax^2 + bx + c to data
# and then rank points according to yhat
# and return the fraction in the top-K that have a = 1
fit_and_evaluate_topk <- function(df, K, split = F) {
  if (split) {
    df_train <- filter(df, split == "train")
    df_holdout <- filter(df, split == "holdout")
  } else {
    df_train <- df
    df_holdout <- df
  }
  model <- lm(y ~ x + I(x^2), data = df_train)

  df_holdout %>%
    add_predictions(model) %>%
    top_n(K, pred) %>%
    summarize(frac_a_in_top_k = mean(a)) %>%
    pull()
}

# a function to create healthcare data with faked outcomes
# remove the original 'dem_age_band.*' buckets from the original data and
# fake and join continuous age back in as 'dem_age'
# then overwrite the three outcomes
# log_cost_t -> parabola at -1
# gagne_sum_t -> parabola at b (passed as parameter)
# log_cost_avoidable_t -> random noise
generate_semi_synthetic_healthcare_data <- function(df_orig, b, standardize = F) {
  # convert each person's age from an indicator for a discrete age bucket
  # to a continuous age by sampling uniformly within the specified age bucket
  # note: there are some people who are assigned two different age bins
  # and there are some people who are missing an age entirely
  # so deal with that by assuming that the youngest recorded age is correct
  # and fill in missing values with the youngest age bucket
  df_age <- df_orig %>%
    select(index, matches('dem_age_band')) %>%
    pivot_longer(names_to = "variable", values_to = "value", matches('dem_age_band')) %>%
    #filter(value == 1) %>%
    group_by(index) %>%
    arrange(index, desc(value), variable) %>%
    slice(1) %>%
    ungroup() %>%
    extract(variable, c("age_min","age_max"), "dem_age_band_([0-9]+)-([0-9]+)_tm1") %>%
    mutate(age_min = as.numeric(ifelse(is.na(age_min), 75, age_min)),
           age_max = as.numeric(ifelse(is.na(age_max), 84, age_max))) %>%
    mutate(dem_age = round(runif(n(), age_min, age_max)),
           dem_age_sq = dem_age^2) %>%
    select(index, dem_age, dem_age_sq)
  
  # overwrite the three outcomes
  # y1 -> log_cost_t
  # y2 -> gagne_sum_t
  # and set log_cost_avoidable_t to a convex up parabola at x = 0
  set.seed(134)
  df_new <- df_orig %>%
    select(-matches('dem_age_band')) %>%
    left_join(df_age) %>%
    mutate(dem_race_black = rbinom(n(), 1, ifelse(dem_age >= 45 & dem_age <= 54, .22, .11))) %>%
    mutate(log_cost_t = 1 - 0.25 * (as.numeric(scale(dem_age)) + 1)^2 + 0.01*rnorm(n()),
           gagne_sum_t = 1 - 0.25 * (as.numeric(scale(dem_age)) - b)^2 + 0.01*rnorm(n()),
           log_cost_avoidable_t = 0.1*rnorm(n()))
           #log_cost_avoidable_t = as.numeric(scale(dem_age))^2 + 0.01*rnorm(n()))
  
  if (standardize) {
    df_new <- df_new %>%
      mutate(log_cost_t = as.numeric(scale(log_cost_t)),
             log_cost_avoidable_t = as.numeric(scale(log_cost_avoidable_t)),
             gagne_sum_t = as.numeric(scale(gagne_sum_t)))
  }
}

################################################################################
# Generate semi-synthetic versions of the healthcare data w/ parabolic outcomes
################################################################################

# read in the original data
df_orig <- read_csv('data/dissecting_bias_dataset_three_outcomes_limited_features_train_and_holdout.csv') %>%
  mutate(index = row_number())

# set some parameters and a data frame to store results
set.seed(134)
bs <- seq(-1.1, 1.1, by = 0.1)
alphas <- seq(0, 1, by = 0.1)
results <- data.frame()

# loop over b, which sets the center of gagne_sum_t
for (b in bs) {
  print(paste("b:", b), sep = " ")
  # generate semi-synthetic data using this b value
  # select only the few columns we need here
  df <- generate_semi_synthetic_healthcare_data(df_orig, b = b, standardize = T) 

  write_csv(df, file=sprintf('data/dissecting_bias_dataset_semisynthetic_b=%.1f.csv', b))
  
  df <- df %>%
    select(log_cost_t, gagne_sum_t, log_cost_avoidable_t,
           a = dem_race_black, x = dem_age, split)
  
  # fit the log_cost_t model and compute frac a = 1 among top-k predictions
  cost_frac_a_in_topk <- df %>%
    mutate(y = log_cost_t) %>%
    fit_and_evaluate_topk(0.03*nrow(df), split = T)
  results <- bind_rows(results,
                       data.frame(b = b, outcome = 'log_cost_t',
                                  alpha1 = NA, alpha2 = NA, alpha3 = NA,
                                  frac_a_in_topk = cost_frac_a_in_topk))
  print(paste("cost_frac_a_in_topk:", cost_frac_a_in_topk), sep = " ")

  # fit the gagne_sum_t model and compute frac a = 1 among top-k predictions
  gagne_frac_a_in_topk <- df %>%
    mutate(y = gagne_sum_t) %>%
    fit_and_evaluate_topk(0.03*nrow(df), split = T)
  results <- bind_rows(results,
                       data.frame(b = b, outcome = 'gagne_sum_t',
                                  alpha1 = NA, alpha2 = NA, alpha3 = NA,
                                  frac_a_in_topk = gagne_frac_a_in_topk))
  print(paste("gagne_frac_a_in_topk:", gagne_frac_a_in_topk), sep = " ")

  # fit the avoidable cost model and compute frac a = 1 among top-k predictions
  avoidable_cost_frac_a_in_topk <- df %>%
    mutate(y = log_cost_avoidable_t) %>%
    fit_and_evaluate_topk(0.03*nrow(df), split = T)
  print(paste("avoidable_cost_frac_a_in_topk:", avoidable_cost_frac_a_in_topk), sep = " ")

  results <- bind_rows(results,
                       data.frame(b = b, outcome = 'log_cost_avoidable_t',
                                  alpha1 = NA, alpha2 = NA, alpha3 = NA,
                                  frac_a_in_topk = avoidable_cost_frac_a_in_topk))

  # grid search over alpha to find the best index model
  # yim = alpha1 * cost + alpha2 * gagne + (1 - alpha1 - alpha2) * avoidable
  for (alpha1 in alphas) {
    for (alpha2 in alphas) {
      # skip if alpha1 + alpha2 > 1
      if (alpha1 + alpha2 > 1) {
        next
      }

      yim_frac_a_in_topk <- df %>%
        mutate(y = alpha1 * log_cost_t + alpha2 * gagne_sum_t +
                   (1 - alpha1 - alpha2) * log_cost_avoidable_t) %>%
        fit_and_evaluate_topk(0.03*nrow(df))

      results <- bind_rows(results,
                           data.frame(b = b, outcome = 'yim',
                                      alpha1 = alpha1, alpha2 = alpha2,
                                      alpha3 = (1 - alpha1 - alpha2),
                                      frac_a_in_topk = yim_frac_a_in_topk))
    }
  }
  
  b_curr <- b
  index_model_frac_a_in_topk <- results %>%
    group_by(b, outcome) %>%
    arrange(b, outcome, alpha1, alpha2) %>%
    mutate(alpha_rank = rank(desc(frac_a_in_topk), ties.method = 'first')) %>%
    filter(b == b_curr, alpha_rank == 1, outcome == "yim")
  print("best index model:")
  print(index_model_frac_a_in_topk)
}

plot_data <- results %>%
  group_by(b, outcome) %>%
  arrange(b, outcome, alpha1, alpha2) %>%
  mutate(alpha_rank = rank(desc(frac_a_in_topk), ties.method = 'first')) %>%
  filter(alpha_rank == 1) 
plot_data %>%
  ggplot(aes(x = b, y = frac_a_in_topk, color = outcome)) +
  geom_point(aes(shape = outcome), size = 2) +
  geom_line(aes(linetype = outcome)) +
  scale_y_continuous(label = percent) +
  labs(x = 'Offset of gagne_sum_t peak',
       y = 'Percent of group a in top-k',
       color = 'Target', linetype = 'Target', shape = 'Target')
