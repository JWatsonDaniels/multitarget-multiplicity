library(tidyverse)
library(scales)
library(patchwork)

theme_set(theme_bw())

table2 <- read_csv('results/dissecting_bias_index_model_concentration_metric.csv')

table2_long <- table2 %>%
  pivot_longer(-predictor, names_to = 'variable', values_to = 'value') %>%
  mutate(is_se = grepl('SE$', variable)) %>%
  mutate(variable = gsub('Race black', 'Black patients', variable))

table2_estimates <- table2_long %>%
  filter(!is_se) %>%
  rename(estimate = value) %>%
  select(-is_se)
table2_SEs <- table2_long %>%
  filter(is_se) %>%
  rename(se = value) %>%
  select(-is_se) %>%
  mutate(variable = gsub(' SE$', '', variable))

variable_order <- c('Total costs', 'Avoidable costs', 'Active chronic conditions', 'Index model', 'Black patients')
variable_order_newlines <- gsub(' ', '\n', variable_order)

plot_data <- inner_join(table2_estimates, table2_SEs) %>%
  filter(!grepl('^Best-worst', predictor)) %>%
  mutate(variable = factor(variable, levels = variable_order),
         predictor = gsub(' ', '\n', predictor),
         predictor = factor(predictor, levels = variable_order_newlines),
         is_index = grepl('Index.*model', predictor))

p1 <- plot_data %>%
  filter(variable != 'Black patients') %>%
  ggplot(aes(x = predictor, y = estimate)) +
  geom_bar(aes(color = variable, fill = variable, alpha = is_index), stat = "identity") +
  scale_y_continuous(label = percent) +
  facet_wrap(~ variable, nrow = 1) +
  theme(legend.position = "none") +
  labs(x = '\nTarget variable',
       y = 'Percent of outcome covered\nby highest-risk patients')

p2 <- plot_data %>%
  filter(variable == 'Black patients') %>%
  ggplot(aes(x = predictor, y = estimate)) +
  geom_bar(aes(alpha = is_index), stat = "identity", color = 'black') +
  scale_y_continuous(label = percent) +
  facet_wrap(~ variable, nrow = 1) +
  theme(legend.position = "none") +
  labs(x = '\nTarget variable',
       y = 'Percent of highest-risk\npatients who are Black')

p1 + p2 +
  plot_annotation(tag_levels = 'A') +
  plot_layout(widths = c(3, 1))

ggsave(filename = 'figures/healthcare_concentration_table_as_bar_plot.pdf', width = 10, height = 3)
ggsave(filename = 'figures/healthcare_concentration_table_as_bar_plot.png', width = 10, height = 3)