library(tidyverse)
library(scales)
library(modelr)
library(patchwork)

theme_set(theme_bw())

################################################################################
# Define functions
################################################################################

# a function to fit a parabola y ~ ax^2 + bx + c to data
# and then rank points according to yhat
# and return the fraction in the top-K that have a = 1
fit_and_evaluate_topk <- function(df, K) {
  model <- lm(y ~ x + I(x^2), data = df)
  df %>%
    add_predictions(model) %>%
    top_n(K, pred) %>%
    summarize(frac_a_in_top_k = mean(a)) %>%
    pull()
}

plot_outcomes <- function(N, b, delta, y2_color) {
  fname <- sprintf('data/dissecting_bias_dataset_semisynthetic_b=%.1f.csv', b)
  df <- read_csv(fname) %>%
    rename(x = dem_age, a = dem_race_black, y1 = log_cost_t, y2 = gagne_sum_t)
  
  alphas <- seq(0, 1, by = 0.05)
  results <- data.frame()
  for (alpha in alphas) {
    yim_frac_a_in_topk <- df %>%
      mutate(y = alpha * y1 + (1 - alpha) * y2) %>%
      fit_and_evaluate_topk(N*0.03)
    
    results <- bind_rows(results,
                         data.frame(b = b, outcome = 'yim', alpha = alpha,
                                    frac_a_in_topk = yim_frac_a_in_topk))
  }
  
  best_alpha <- results %>%
    group_by(b, outcome) %>%
    arrange(b, outcome, alpha) %>%
    mutate(alpha_rank = rank(desc(frac_a_in_topk), ties.method = 'last')) %>%
    filter(alpha_rank == 1) %>%
    pull(alpha)
  
  #outcome_labels <- c(yim = "Index model",
  #                    y1 = "Total costs",
  #                    y2 = "Active chronic conditions")
  plot_data <- df %>%
    mutate(yim = best_alpha * y1 + (1 - best_alpha) * y2) %>%
    select(x, yim, y1, y2) %>%
    pivot_longer(names_to = "variable", values_to = "value", -x)
  annotation_data <- data.frame(xmin = 45, xmax = 54,
                                ymin = -Inf, ymax = Inf,
                                label = "Highest concentration\nof black patients")
  outcome_labels <- c(
    expression(hat(y)[IM]),
    expression(hat(y)^{(1)}),
    expression(hat(y)^{(2)})
  )
  plot_data %>%
    filter(variable != "a") %>%
    mutate(variable = factor(variable, levels = c('yim','y1','y2'))) %>%
    #mutate(variable = recode_factor(variable, !!!outcome_labels)) %>%
    ggplot() +
    #geom_point() +
    geom_rect(data = annotation_data, aes(xmin = xmin, xmax = xmax,
                                          ymin = ymin, ymax = ymax,
                                          fill = "Protected group"),
    #annotate("rect", xmin = -delta, xmax = delta, ymin = -Inf, ymax = Inf,
              alpha = 0.1, color = "NA") +
    geom_smooth(aes(x = x, y = value, color = variable, shape = variable, linetype = variable),
                method = lm, formula = "y ~ x + I(x^2)", se = F) +
    scale_color_manual(values = c('#54278f', 'red', y2_color), labels = outcome_labels) +
    scale_shape_manual(values = c(4, 1, 3), labels = outcome_labels) +
    scale_linetype_manual(values = c('solid', 'dotted', 'dashed'), labels = outcome_labels) +
    scale_fill_manual('', values = 'green') +
    theme(legend.position = "none") +
    labs(color = "",
         shape = "",
         linetype = "",
         fill = 'Protected group',
         x = "Age",
         y = "Target"
         #subtitle = "y1 = 1 - 1/4*(x + 1)^2; y2 = 1 - 1/4*(x - b)^2"
    ) #; yim = 0*y1 + 1*y2")
}


################################################################################
# Plot the left panel (Figure 2A)
################################################################################

set.seed(23)
N <- 200
delta = 0.25


bs <- c(-0.5, -0.2, 0, 0.2, 0.5)
df <- map_df(bs, function(b) { 
  fname <- sprintf('data/dissecting_bias_dataset_semisynthetic_b=%.1f.csv', b)
  data.frame(b = b, read_csv(fname)) %>%
    rename(x = dem_age, y1 = log_cost_t, y2 = gagne_sum_t)
}) %>%
  mutate(b = factor(b, levels = bs))
annotation_data <- data.frame(xmin = 45, xmax = 54,
                              ymin = -Inf, ymax = Inf,
                              label = "Protected group")
(p1 <- ggplot(df) +
    geom_rect(data = annotation_data, aes(xmin = xmin, xmax = xmax,
                                          ymin = ymin, ymax = ymax,
                                          fill = "Protected group"),
              alpha = 0.1, color = "NA") +
    #annotate("rect", xmin = -delta, xmax = delta, ymin = -Inf, ymax = Inf,
    #         fill = "green", alpha = 0.1, color = "NA") +
    geom_smooth(aes(x = x, y = y2, color = b, group = b),
                method = lm, formula = "y ~ x + I(x^2)", se = F, linetype = 'dashed') +
    scale_shape_manual(values = 4) +
    scale_color_manual(values = c(#"#eff3ff",
                                  "#c6dbef", "#9ecae1","#6baed6","#4292c6","#2171b5"#,
                                  #"#084594"
                                  )) +
    #scale_color_brewer(type = "seq", palette = 1) +
    scale_fill_manual('', values = 'green') +
    theme(legend.position = "bottom") +
    guides(fill  = guide_legend(order = 1),
           color = guide_legend(order = 2)) +
    labs(x = 'Age', y = expression(hat(y)^{(2)}), color = "b")
)

ggsave(filename = 'figures/healthcare_semisynthetic_data_left.pdf', width = 4, height = 4)


################################################################################
# Plot the middle panel (Figure 2B)
################################################################################

(p2 <- plot_outcomes(N, -0.5, delta, "#c6dbef") +
    #theme(legend.position = "none") +
    labs(title = 'i. b = -0.5')
)
(p3 <- plot_outcomes(N, -0.2, delta, "#9ecae1") +
    #theme(legend.position = "none") +
    labs(y = '', title = 'ii. b = -0.2')
)
(p4 <- plot_outcomes(N, 0.5, delta, "#2171b5") +
    #theme(legend.position = "none") +
    labs(y = '', title = 'iii. b = 0.5')
)

p2 + p3 + p4 + plot_layout(guides = "collect") & theme(legend.position = "bottom")
ggsave(filename = 'figures/healthcare_semisynthetic_data_middle.pdf', width = 8, height = 4)

################################################################################
# Plot the right panel (Figure 2C)
################################################################################

results <- read_csv('results/dissecting_bias_dataset_semisynthetic_tuneK30_tuneN300_allbvalues.csv')

outcome_labels <- c(
  frac_index_model = 'Index\nmodel',
  frac_log_cost_t = 'Total\ncost',
  frac_gagne_t = 'Active\nchronic\nconditions'
)
annotation_data <- data.frame(
  b = c(-0.5,-0.25,0.5),
  y = c(0.12, 0.18, 0.22),
  label = c('i','ii','iii')
)
plot_data <- results %>% 
  pivot_longer(names_to = "variable", values_to = "value", starts_with('frac')) %>% 
  mutate(variable = factor(variable, levels = names(outcome_labels)),
         variable = recode_factor(variable, !!!outcome_labels),
         se = sqrt(value * (1 - value))/sqrt(top_K))
outcome_labels <- c(
  expression(hat(y)[IM]),
  expression(hat(y)^{(1)}),
  expression(hat(y)^{(2)})
)
(p5 <- ggplot(plot_data) +
  geom_point(aes(x = b, y = value, color = variable, linetype = variable, shape = variable)) + 
  annotate("rect", xmin = -1, xmax = 1, ymin = .108 - 0.0141, ymax = .108 + 0.0141, fill = "red", color = "NA", alpha=0.2) +
  geom_smooth(aes(x = b, y = value, color = variable, linetype = variable, shape = variable, fill = variable)) + 
  scale_color_manual(values = c('#54278f', 'red', '#2171b5'), labels = outcome_labels) +
  scale_fill_manual(values = c('#54278f', 'red', '#2171b5'), labels = outcome_labels) +
  scale_shape_manual(values = c(4, 1, 3), labels = outcome_labels) +
  scale_linetype_manual(values = c('solid', 'dotted', 'dashed'), labels = outcome_labels) +
  scale_y_continuous(label = percent, lim = c(0,.25)) +
  geom_label(data = annotation_data, aes(b, y, label = label), show.legend = F, size = 2) +
  labs(x = 'b',
       y = 'Percent of highest-risk\npatients in protected group',
       color = 'Target', shape = 'Target', linetype = 'Target', fill = 'Target') +
  theme(legend.position = "bottom")
)

ggsave(filename = 'figures/healthcare_semisynthetic_data_right.pdf', width = 4, height = 4)

