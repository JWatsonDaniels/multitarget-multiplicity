library(tidyverse)
library(scales)

theme_set(theme_bw())

################################################################################
# Figure 2B
################################################################################

variable_order <- c('Total costs', 'Avoidable costs', 'Active chronic conditions', 'Index model', 'Black patients')
variable_order_newlines <- gsub(' ', '\n', variable_order)

# read single target ambiguities
ambiguities <- read_csv('results/dissecting_bias_single_target_ambiguities_with_limited_features.csv')

# note that index model ambiguity is hard-coded as 776/1000 below
ambiguities %>%
  #mutate(nk = 'kappa') %>%
  pivot_longer(names_to = "Target", values_to = "value", c(`Total costs`, `Avoidable costs`, `Active chronic conditions`)) %>%
  mutate(Target = gsub(' ', '\n', Target),
         Target = factor(Target, levels = variable_order_newlines)) %>%
  ggplot(aes(x = epsilon, y = value, color = Target)) +
  geom_line() +
  geom_point() +
  geom_hline(yintercept = 56 / 1000, linetype = 'dashed') +
  annotate("point", x = 0, y = 56 / 1000, shape = 'x', size = 4) +
  labs(x = expression("Maximum relative tolerance on MSE"~(epsilon)),
       y = 'Ambiguity',
       title = expression(paste("Top-", kappa, " Ambiguity (all) (n = 1000, ", kappa, " = 40)"))) +
  #theme(legend.position = "bottom") +
  #theme(legend.position = c(0.85, 0.75), legend.background = element_blank()) +
  scale_x_continuous(lim = c(0, 2.5/100), expand = c(0.0005, 0.0005)) +
  scale_y_continuous(label = percent)
#facet_wrap(~ nk, labeller = label_parsed) +

ggsave(filename = 'figures/healthcare_single_vs_multi_target_ambiguity.pdf', width = 5.5, height = 4)

# hard-coded results from experiments below
data.frame(k = c(10,20,40,60,100,500,600), k_stable = c(1,6,15,27,52,382,481)) %>% 
  ggplot(aes(x = k/1000, y = k_stable/k)) + 
  geom_point() + 
  geom_line() + 
  labs(x = expression(paste('Percent of points in top-', kappa ,' set (', kappa, '/n)')), 
       y = expression(paste("Percent of stable points within the top-", kappa, " set")),
       title = "Stability (selected, n = 1000)") + 
  scale_y_continuous(label = percent) +
  scale_x_continuous(label = percent)

ggsave(filename = 'figures/healthcare_multi_target_stability.pdf', width = 5.5, height = 4)
