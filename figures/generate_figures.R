#!/usr/bin/env Rscript
# ==============================================================================
# generate_figures.R
# Publication-quality figures for the mlx-snn arXiv paper
#
# Usage: Rscript figures/generate_figures.R
# (run from the mlx-snn project root)
#
# Produces 8 PDF figures in figures/ directory.
# ==============================================================================

options(warn = -1)

library(tidyverse)
library(patchwork)
library(scales)
library(RColorBrewer)
library(viridis)

# ==============================================================================
# Global style settings
# ==============================================================================

# Academic color palette
col_mlxsnn   <- "#2166AC"   # Steel blue
col_snntorch <- "#D6604D"   # Muted red-orange
col_ref_line <- "grey50"

# Custom theme for all plots
theme_paper <- function(base_size = 11) {
  theme_bw(base_size = base_size) +
    theme(
      text             = element_text(family = "serif"),
      plot.title       = element_text(face = "bold", size = rel(1.1), hjust = 0.5),
      plot.subtitle    = element_text(size = rel(0.85), hjust = 0.5, color = "grey40"),
      strip.text       = element_text(face = "bold", size = rel(0.95)),
      strip.background = element_rect(fill = "grey95", color = NA),
      panel.grid.minor = element_blank(),
      legend.position  = "bottom",
      legend.title     = element_text(face = "bold"),
      axis.title       = element_text(face = "bold"),
      plot.margin      = margin(8, 8, 8, 8)
    )
}

# Output directory
fig_dir <- "figures"
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# Data loading utilities
# ==============================================================================

dir_original <- "experiments/results_v04"
dir_fixed    <- "experiments/results_v04_fixed"
dir_5seeds   <- "experiments/results_v04_5seeds"

#' Safely read a CSV file; return NULL if it does not exist.
safe_read <- function(path) {
  if (file.exists(path)) {
    df <- read_csv(path, show_col_types = FALSE)
    return(df)
  }
  return(NULL)
}

#' Load and merge data for a given experiment/config across all 3 directories.
#' @param exp_config  e.g. "exp1_Leaky"
#' @param dirs        Named list of directories to search
#' @param filenames   Named list mapping directory key -> filename (without dir prefix).
#'                    If NULL, use the standard naming pattern.
load_config <- function(exp_config, dirs = NULL, filenames = NULL) {
  if (is.null(dirs)) {
    dirs <- list(
      original = dir_original,
      fixed    = dir_fixed,
      extra    = dir_5seeds
    )
  }

  frames <- list()
  for (key in names(dirs)) {
    if (!is.null(filenames) && key %in% names(filenames)) {
      fn <- filenames[[key]]
    } else {
      fn <- paste0("curves_", exp_config, ".csv")
    }
    path <- file.path(dirs[[key]], fn)
    df <- safe_read(path)
    if (!is.null(df)) {
      frames[[key]] <- df
    }
  }

  if (length(frames) == 0) return(NULL)
  bind_rows(frames) %>% distinct(seed, epoch, .keep_all = TRUE)
}

# ==============================================================================
# Load all experiment data
# ==============================================================================

cat("Loading experiment data...\n")

# --- Experiment 1: Neuron types ---
# Leaky and Synaptic: original dir only (fixed/5seeds may have extras)
exp1_leaky    <- load_config("exp1_Leaky")
exp1_synaptic <- load_config("exp1_Synaptic")

# RLeaky and RSynaptic: use V0.5_fixed from the fixed dir as canonical,
# and also try to pick up extra seeds from 5seeds dir
exp1_rleaky <- load_config(
  "exp1_RLeaky_V0.5_fixed",
  dirs = list(fixed = dir_fixed, extra = dir_5seeds),
  filenames = list(extra = "curves_exp1_RLeaky_V0.5.csv")
)
exp1_rsynaptic <- load_config(
  "exp1_RSynaptic_V0.5_fixed",
  dirs = list(fixed = dir_fixed, extra = dir_5seeds),
  filenames = list(extra = "curves_exp1_RSynaptic_V0.5.csv")
)

# For Figure 8 (recurrent V comparison), also load V=0.1 learn and original V=1.0
exp1_rleaky_v01   <- load_config(
  "exp1_RLeaky_V0.1_learn",
  dirs = list(fixed = dir_fixed, extra = dir_5seeds)
)
exp1_rsynaptic_v01 <- load_config(
  "exp1_RSynaptic_V0.1_learn",
  dirs = list(fixed = dir_fixed, extra = dir_5seeds)
)
exp1_rleaky_orig   <- load_config(
  "exp1_RLeaky",
  dirs = list(original = dir_original)
)
exp1_rsynaptic_orig <- load_config(
  "exp1_RSynaptic",
  dirs = list(original = dir_original)
)

# --- Experiment 2: Learnable parameters ---
exp2_baseline     <- load_config("exp2_baseline")
exp2_learn_beta   <- load_config("exp2_learn_beta")
exp2_learn_thresh <- load_config("exp2_learn_thresh")
exp2_learn_both   <- load_config("exp2_learn_both")

# --- Experiment 3: Surrogate gradients ---
exp3_fast_sigmoid <- load_config("exp3_fast_sigmoid")
exp3_arctan       <- load_config("exp3_arctan")
exp3_sigmoid      <- load_config("exp3_sigmoid")
# Triangular and straight_through: use fixed dir as canonical (fixes bugs)
exp3_triangular <- load_config(
  "exp3_triangular",
  dirs = list(fixed = dir_fixed, extra = dir_5seeds)
)
exp3_straight_through <- load_config(
  "exp3_straight_through",
  dirs = list(fixed = dir_fixed, extra = dir_5seeds)
)

# --- Experiment 4: Loss functions ---
exp4_ce_rate      <- load_config("exp4_ce_rate_loss")
exp4_ce_count     <- load_config("exp4_ce_count_loss")
exp4_mse_membrane <- load_config("exp4_mse_membrane_loss")

# --- Experiment 5: Encoding methods ---
exp5_rate    <- load_config("exp5_rate")
exp5_latency <- load_config("exp5_latency")
exp5_delta   <- load_config("exp5_delta")

# ==============================================================================
# Helper: compute final accuracy summary for a config
# ==============================================================================

final_summary <- function(df, config_name) {
  if (is.null(df) || nrow(df) == 0) {
    return(tibble(config = config_name, mean_acc = NA_real_,
                  sd_acc = NA_real_, n_seeds = 0L))
  }
  df %>%
    group_by(seed) %>%
    slice_max(epoch, n = 1) %>%
    ungroup() %>%
    summarise(
      config   = config_name,
      mean_acc = mean(test_acc) * 100,
      sd_acc   = sd(test_acc) * 100,
      n_seeds  = n_distinct(seed),
      mean_time = mean(epoch_time, na.rm = TRUE)
    )
}

# ==============================================================================
# snnTorch baseline results (V100)
# ==============================================================================

snntorch_baselines <- tribble(
  ~experiment, ~config,              ~mean_acc, ~sd_acc,
  "Exp 1",     "Leaky",              97.38,     0.09,
  "Exp 1",     "RLeaky (V=0.5)",     75.10,     2.51,
  "Exp 1",     "Synaptic",           96.08,     0.05,
  "Exp 1",     "RSynaptic (V=0.5)",  58.68,     14.52,
  "Exp 2",     "baseline",           97.38,     0.09,
  "Exp 2",     "learn_beta",         97.06,     0.08,
  "Exp 2",     "learn_thresh",       97.35,     0.03,
  "Exp 2",     "learn_both",         97.27,     0.10,
  "Exp 3",     "fast_sigmoid",       96.82,     0.08,
  "Exp 3",     "arctan",             97.38,     0.09,
  "Exp 3",     "sigmoid",            9.91,      0.16,
  "Exp 3",     "triangular",         72.73,     7.02,
  "Exp 3",     "straight_through",   69.32,     0.82,
  "Exp 4",     "ce_rate",            97.38,     0.09,
  "Exp 4",     "ce_count",           97.57,     0.12,
  "Exp 4",     "mse_membrane",       96.99,     0.11,
  "Exp 5",     "rate",               97.38,     0.09,
  "Exp 5",     "latency",            94.48,     0.04,
  "Exp 5",     "delta",              82.45,     0.17
) %>%
  mutate(framework = "snnTorch (V100)")

# ==============================================================================
# Compute mlx-snn final summaries
# ==============================================================================

mlx_summaries <- bind_rows(
  final_summary(exp1_leaky,            "Leaky")            %>% mutate(experiment = "Exp 1"),
  final_summary(exp1_rleaky,           "RLeaky (V=0.5)")   %>% mutate(experiment = "Exp 1"),
  final_summary(exp1_synaptic,         "Synaptic")         %>% mutate(experiment = "Exp 1"),
  final_summary(exp1_rsynaptic,        "RSynaptic (V=0.5)") %>% mutate(experiment = "Exp 1"),
  final_summary(exp2_baseline,         "baseline")         %>% mutate(experiment = "Exp 2"),
  final_summary(exp2_learn_beta,       "learn_beta")       %>% mutate(experiment = "Exp 2"),
  final_summary(exp2_learn_thresh,     "learn_thresh")     %>% mutate(experiment = "Exp 2"),
  final_summary(exp2_learn_both,       "learn_both")       %>% mutate(experiment = "Exp 2"),
  final_summary(exp3_fast_sigmoid,     "fast_sigmoid")     %>% mutate(experiment = "Exp 3"),
  final_summary(exp3_arctan,           "arctan")           %>% mutate(experiment = "Exp 3"),
  final_summary(exp3_sigmoid,          "sigmoid")          %>% mutate(experiment = "Exp 3"),
  final_summary(exp3_triangular,       "triangular")       %>% mutate(experiment = "Exp 3"),
  final_summary(exp3_straight_through, "straight_through") %>% mutate(experiment = "Exp 3"),
  final_summary(exp4_ce_rate,          "ce_rate")          %>% mutate(experiment = "Exp 4"),
  final_summary(exp4_ce_count,         "ce_count")         %>% mutate(experiment = "Exp 4"),
  final_summary(exp4_mse_membrane,     "mse_membrane")     %>% mutate(experiment = "Exp 4"),
  final_summary(exp5_rate,             "rate")             %>% mutate(experiment = "Exp 5"),
  final_summary(exp5_latency,          "latency")          %>% mutate(experiment = "Exp 5"),
  final_summary(exp5_delta,            "delta")            %>% mutate(experiment = "Exp 5")
) %>%
  mutate(framework = "mlx-snn (M3 Max)") %>%
  filter(!is.na(mean_acc))

cat(sprintf("  Loaded %d mlx-snn configs across %d experiments.\n",
            nrow(mlx_summaries), n_distinct(mlx_summaries$experiment)))

# ==============================================================================
# FIGURE 1: Learning curves for Experiment 1 (Neuron Types)
# ==============================================================================

cat("Generating Figure 1: Experiment 1 learning curves...\n")

build_curves_df <- function(df, label) {
  if (is.null(df)) return(NULL)
  df %>% mutate(config = label, test_acc_pct = test_acc * 100)
}

exp1_curves <- bind_rows(
  build_curves_df(exp1_leaky,     "Leaky"),
  build_curves_df(exp1_rleaky,    "RLeaky (V=0.5)"),
  build_curves_df(exp1_synaptic,  "Synaptic"),
  build_curves_df(exp1_rsynaptic, "RSynaptic (V=0.5)")
) %>%
  mutate(config = factor(config, levels = c("Leaky", "Synaptic",
                                            "RLeaky (V=0.5)", "RSynaptic (V=0.5)")))

exp1_summary <- exp1_curves %>%
  group_by(config, epoch) %>%
  summarise(
    mean_acc = mean(test_acc_pct),
    sd_acc   = sd(test_acc_pct),
    .groups  = "drop"
  ) %>%
  mutate(
    lo = pmax(mean_acc - sd_acc, 0),
    hi = pmin(mean_acc + sd_acc, 100)
  )

p1 <- ggplot() +
  # Individual seed traces
  geom_line(data = exp1_curves,
            aes(x = epoch, y = test_acc_pct, group = interaction(config, seed)),
            alpha = 0.25, linewidth = 0.4, color = "grey40") +
  # Mean +/- std ribbon
  geom_ribbon(data = exp1_summary,
              aes(x = epoch, ymin = lo, ymax = hi),
              fill = col_mlxsnn, alpha = 0.2) +
  # Mean line
  geom_line(data = exp1_summary,
            aes(x = epoch, y = mean_acc),
            color = col_mlxsnn, linewidth = 1) +
  facet_wrap(~ config, nrow = 2, scales = "free_y") +
  labs(
    title    = "Experiment 1: Neuron Type Comparison",
    subtitle = "Test accuracy over training epochs (mlx-snn on M3 Max)",
    x = "Epoch", y = "Test Accuracy (%)"
  ) +
  theme_paper() +
  theme(legend.position = "none")

ggsave(file.path(fig_dir, "fig_learning_curves_exp1.pdf"), p1,
       width = 7, height = 5, device = cairo_pdf)

# ==============================================================================
# FIGURE 2: Grouped bar chart comparing final accuracy across all experiments
# ==============================================================================

cat("Generating Figure 2: Accuracy comparison (all experiments)...\n")

# Combine mlx-snn and snnTorch data
all_results <- bind_rows(
  mlx_summaries %>% select(experiment, config, mean_acc, sd_acc, framework),
  snntorch_baselines %>% select(experiment, config, mean_acc, sd_acc, framework)
)

# Set consistent factor ordering within each experiment
config_order <- c(
  # Exp 1
  "Leaky", "Synaptic", "RLeaky (V=0.5)", "RSynaptic (V=0.5)",
  # Exp 2
  "baseline", "learn_beta", "learn_thresh", "learn_both",
  # Exp 3
  "fast_sigmoid", "arctan", "sigmoid", "triangular", "straight_through",
  # Exp 4
  "ce_rate", "ce_count", "mse_membrane",
  # Exp 5
  "rate", "latency", "delta"
)
all_results <- all_results %>%
  mutate(config = factor(config, levels = config_order))

p2 <- ggplot(all_results,
             aes(x = config, y = mean_acc, fill = framework)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.65, color = "white", linewidth = 0.2) +
  geom_errorbar(aes(ymin = mean_acc - sd_acc, ymax = mean_acc + sd_acc),
                position = position_dodge(width = 0.7), width = 0.2, linewidth = 0.4) +
  geom_hline(yintercept = 90, linetype = "dashed", color = col_ref_line, linewidth = 0.4) +
  facet_wrap(~ experiment, scales = "free_x", nrow = 1) +
  scale_fill_manual(
    values = c("mlx-snn (M3 Max)" = col_mlxsnn, "snnTorch (V100)" = col_snntorch),
    name   = "Framework"
  ) +
  scale_y_continuous(limits = c(0, 105), breaks = seq(0, 100, 20)) +
  labs(
    title = "Final Test Accuracy: mlx-snn vs snnTorch",
    x     = NULL,
    y     = "Test Accuracy (%)"
  ) +
  theme_paper(base_size = 9) +
  theme(
    axis.text.x = element_text(angle = 35, hjust = 1, size = rel(0.8)),
    legend.position = "bottom"
  )

ggsave(file.path(fig_dir, "fig_accuracy_comparison.pdf"), p2,
       width = 12, height = 5, device = cairo_pdf)

# ==============================================================================
# FIGURE 3: Surrogate gradient functions and their derivatives
# ==============================================================================

cat("Generating Figure 3: Surrogate gradient shapes...\n")

x_vals <- seq(-3, 3, length.out = 500)

# Forward (smooth approximation) and gradient for each surrogate
surrogate_data <- tibble(x = x_vals) %>%
  mutate(
    # Fast sigmoid
    `fast_sigmoid_fwd`  = 1 / (1 + exp(-25 * x)),
    `fast_sigmoid_grad` = 25 / (1 + abs(25 * x))^2,
    # Arctan
    `arctan_fwd`  = 0.5 + (1/pi) * atan(2 * pi/2 * x),
    `arctan_grad` = 2 / (2 * (1 + (pi/2 * 2 * x)^2)),
    # Sigmoid
    `sigmoid_fwd`  = 1 / (1 + exp(-x)),
    `sigmoid_grad` = exp(-x) / (1 + exp(-x))^2,
    # Triangular
    `triangular_fwd`  = pmax(0, pmin(1, 0.5 + x)),
    `triangular_grad` = ifelse(abs(x) <= 1, 1.0, 0.0),
    # Straight-through (STE)
    `straight_through_fwd`  = pmax(0, pmin(1, 0.5 + 0.5 * x)),
    `straight_through_grad` = ifelse(abs(x) <= 1, 0.5, 0.0)
  )

# Reshape for plotting
fwd_long <- surrogate_data %>%
  select(x, ends_with("_fwd")) %>%
  pivot_longer(-x, names_to = "surrogate", values_to = "value") %>%
  mutate(surrogate = str_remove(surrogate, "_fwd"))

grad_long <- surrogate_data %>%
  select(x, ends_with("_grad")) %>%
  pivot_longer(-x, names_to = "surrogate", values_to = "value") %>%
  mutate(surrogate = str_remove(surrogate, "_grad"))

surr_colors <- c(
  "fast_sigmoid"     = "#1B9E77",
  "arctan"           = "#D95F02",
  "sigmoid"          = "#7570B3",
  "triangular"       = "#E7298A",
  "straight_through" = "#66A61E"
)

surr_labels <- c(
  "fast_sigmoid"     = "Fast Sigmoid",
  "arctan"           = "Arctan",
  "sigmoid"          = "Sigmoid",
  "triangular"       = "Triangular",
  "straight_through" = "Straight-Through"
)

p3a <- ggplot(fwd_long, aes(x = x, y = value, color = surrogate)) +
  geom_line(linewidth = 0.8) +
  geom_vline(xintercept = 0, linetype = "dotted", color = "grey60") +
  scale_color_manual(values = surr_colors, labels = surr_labels, name = "Surrogate") +
  labs(
    title = "(a) Surrogate Forward Pass",
    x     = expression(V[mem] - V[thr]),
    y     = "Output"
  ) +
  coord_cartesian(ylim = c(-0.05, 1.05)) +
  theme_paper() +
  theme(legend.position = "none")

p3b <- ggplot(grad_long, aes(x = x, y = value, color = surrogate)) +
  geom_line(linewidth = 0.8) +
  geom_vline(xintercept = 0, linetype = "dotted", color = "grey60") +
  scale_color_manual(values = surr_colors, labels = surr_labels, name = "Surrogate") +
  labs(
    title = "(b) Surrogate Gradient",
    x     = expression(V[mem] - V[thr]),
    y     = "Gradient magnitude"
  ) +
  theme_paper() +
  theme(legend.position = "bottom")

p3 <- p3a + p3b + plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "bottom")

ggsave(file.path(fig_dir, "fig_surrogate_gradients.pdf"), p3,
       width = 9, height = 4.5, device = cairo_pdf)

# ==============================================================================
# FIGURE 4: Cross-platform dumbbell chart (mlx-snn vs snnTorch)
# ==============================================================================

cat("Generating Figure 4: Cross-platform comparison (dumbbell)...\n")

# Prepare paired data
paired <- inner_join(
  mlx_summaries %>% select(experiment, config, mean_acc, sd_acc) %>% rename(mlx_acc = mean_acc, mlx_sd = sd_acc),
  snntorch_baselines %>% select(experiment, config, mean_acc, sd_acc) %>% rename(snn_acc = mean_acc, snn_sd = sd_acc),
  by = c("experiment", "config")
)

# Order by experiment and config
paired <- paired %>%
  mutate(
    config_label = paste0(experiment, ": ", config),
    config_label = factor(config_label, levels = rev(unique(config_label)))
  )

# Long format for points
paired_long <- paired %>%
  pivot_longer(cols = c(mlx_acc, snn_acc),
               names_to = "framework", values_to = "accuracy") %>%
  mutate(framework = ifelse(framework == "mlx_acc", "mlx-snn (M3 Max)", "snnTorch (V100)"))

p4 <- ggplot() +
  # Connecting segments
  geom_segment(data = paired,
               aes(x = mlx_acc, xend = snn_acc, y = config_label, yend = config_label),
               color = "grey70", linewidth = 0.6) +
  # Points
  geom_point(data = paired_long,
             aes(x = accuracy, y = config_label, color = framework),
             size = 2.5) +
  # Vertical reference lines
  geom_vline(xintercept = c(90, 95), linetype = "dashed", color = "grey80", linewidth = 0.3) +
  scale_color_manual(
    values = c("mlx-snn (M3 Max)" = col_mlxsnn, "snnTorch (V100)" = col_snntorch),
    name   = "Framework"
  ) +
  scale_x_continuous(breaks = seq(0, 100, 10)) +
  labs(
    title = "Cross-Platform Accuracy Comparison",
    subtitle = "mlx-snn (M3 Max) vs snnTorch (V100) for each configuration",
    x = "Test Accuracy (%)", y = NULL
  ) +
  theme_paper() +
  theme(
    panel.grid.major.y = element_line(color = "grey95"),
    legend.position = "bottom"
  )

ggsave(file.path(fig_dir, "fig_cross_platform.pdf"), p4,
       width = 8, height = 7, device = cairo_pdf)

# ==============================================================================
# FIGURE 5: Training speed (seconds/epoch) by experiment config
# ==============================================================================

cat("Generating Figure 5: Training speed per epoch...\n")

# Compute mean epoch time for each config
speed_data <- bind_rows(
  build_curves_df(exp1_leaky,            "Leaky")            %>% mutate(experiment = "Exp 1: Neuron"),
  build_curves_df(exp1_rleaky,           "RLeaky (V=0.5)")   %>% mutate(experiment = "Exp 1: Neuron"),
  build_curves_df(exp1_synaptic,         "Synaptic")         %>% mutate(experiment = "Exp 1: Neuron"),
  build_curves_df(exp1_rsynaptic,        "RSynaptic (V=0.5)") %>% mutate(experiment = "Exp 1: Neuron"),
  build_curves_df(exp2_baseline,         "baseline")         %>% mutate(experiment = "Exp 2: Learnable"),
  build_curves_df(exp2_learn_beta,       "learn_beta")       %>% mutate(experiment = "Exp 2: Learnable"),
  build_curves_df(exp2_learn_thresh,     "learn_thresh")     %>% mutate(experiment = "Exp 2: Learnable"),
  build_curves_df(exp2_learn_both,       "learn_both")       %>% mutate(experiment = "Exp 2: Learnable"),
  build_curves_df(exp3_fast_sigmoid,     "fast_sigmoid")     %>% mutate(experiment = "Exp 3: Surrogate"),
  build_curves_df(exp3_arctan,           "arctan")           %>% mutate(experiment = "Exp 3: Surrogate"),
  build_curves_df(exp3_sigmoid,          "sigmoid")          %>% mutate(experiment = "Exp 3: Surrogate"),
  build_curves_df(exp3_triangular,       "triangular")       %>% mutate(experiment = "Exp 3: Surrogate"),
  build_curves_df(exp3_straight_through, "straight_through") %>% mutate(experiment = "Exp 3: Surrogate"),
  build_curves_df(exp4_ce_rate,          "ce_rate")          %>% mutate(experiment = "Exp 4: Loss"),
  build_curves_df(exp4_ce_count,         "ce_count")         %>% mutate(experiment = "Exp 4: Loss"),
  build_curves_df(exp4_mse_membrane,     "mse_membrane")     %>% mutate(experiment = "Exp 4: Loss"),
  build_curves_df(exp5_rate,             "rate")             %>% mutate(experiment = "Exp 5: Encoding"),
  build_curves_df(exp5_latency,          "latency")          %>% mutate(experiment = "Exp 5: Encoding"),
  build_curves_df(exp5_delta,            "delta")            %>% mutate(experiment = "Exp 5: Encoding")
)

speed_summary <- speed_data %>%
  group_by(experiment, config) %>%
  summarise(
    mean_time = mean(epoch_time, na.rm = TRUE),
    sd_time   = sd(epoch_time, na.rm = TRUE),
    .groups = "drop"
  )

# Order configs within experiments
speed_summary <- speed_summary %>%
  mutate(config = factor(config, levels = config_order))

p5 <- ggplot(speed_summary, aes(x = config, y = mean_time, fill = experiment)) +
  geom_col(width = 0.65, color = "white", linewidth = 0.3) +
  geom_errorbar(aes(ymin = pmax(mean_time - sd_time, 0), ymax = mean_time + sd_time),
                width = 0.2, linewidth = 0.4) +
  facet_wrap(~ experiment, scales = "free_x", nrow = 1) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Training Speed on Apple M3 Max",
    subtitle = "Mean epoch time (seconds) across seeds",
    x = NULL, y = "Time per Epoch (s)"
  ) +
  theme_paper(base_size = 9) +
  theme(
    axis.text.x  = element_text(angle = 35, hjust = 1, size = rel(0.8)),
    legend.position = "none"
  )

ggsave(file.path(fig_dir, "fig_training_speed.pdf"), p5,
       width = 12, height = 4.5, device = cairo_pdf)

# ==============================================================================
# FIGURE 6: Heatmap of final accuracy (all configs x seeds)
# ==============================================================================

cat("Generating Figure 6: Accuracy heatmap (configs x seeds)...\n")

# Collect final-epoch accuracy for each config/seed
collect_final <- function(df, config_name, exp_name) {
  if (is.null(df)) return(NULL)
  df %>%
    group_by(seed) %>%
    slice_max(epoch, n = 1) %>%
    ungroup() %>%
    mutate(config = config_name, experiment = exp_name,
           test_acc_pct = test_acc * 100) %>%
    select(experiment, config, seed, test_acc_pct)
}

heatmap_data <- bind_rows(
  collect_final(exp1_leaky,            "Leaky",              "Exp 1"),
  collect_final(exp1_rleaky,           "RLeaky (V=0.5)",     "Exp 1"),
  collect_final(exp1_synaptic,         "Synaptic",           "Exp 1"),
  collect_final(exp1_rsynaptic,        "RSynaptic (V=0.5)",  "Exp 1"),
  collect_final(exp2_baseline,         "baseline",           "Exp 2"),
  collect_final(exp2_learn_beta,       "learn_beta",         "Exp 2"),
  collect_final(exp2_learn_thresh,     "learn_thresh",       "Exp 2"),
  collect_final(exp2_learn_both,       "learn_both",         "Exp 2"),
  collect_final(exp3_fast_sigmoid,     "fast_sigmoid",       "Exp 3"),
  collect_final(exp3_arctan,           "arctan",             "Exp 3"),
  collect_final(exp3_sigmoid,          "sigmoid",            "Exp 3"),
  collect_final(exp3_triangular,       "triangular",         "Exp 3"),
  collect_final(exp3_straight_through, "straight_through",   "Exp 3"),
  collect_final(exp4_ce_rate,          "ce_rate",            "Exp 4"),
  collect_final(exp4_ce_count,         "ce_count",           "Exp 4"),
  collect_final(exp4_mse_membrane,     "mse_membrane",       "Exp 4"),
  collect_final(exp5_rate,             "rate",               "Exp 5"),
  collect_final(exp5_latency,          "latency",            "Exp 5"),
  collect_final(exp5_delta,            "delta",              "Exp 5")
)

# Create a config label that includes the experiment for grouping on the y-axis
heatmap_data <- heatmap_data %>%
  mutate(
    config_label = paste0(experiment, " | ", config),
    seed_label   = paste0("Seed ", seed)
  )

# Order: by experiment then by config_order within
config_label_order <- heatmap_data %>%
  mutate(config = factor(config, levels = config_order)) %>%
  arrange(experiment, config) %>%
  pull(config_label) %>%
  unique()

heatmap_data <- heatmap_data %>%
  mutate(config_label = factor(config_label, levels = rev(config_label_order)))

p6 <- ggplot(heatmap_data, aes(x = seed_label, y = config_label, fill = test_acc_pct)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.1f", test_acc_pct)),
            size = 2.6, family = "serif") +
  scale_fill_viridis_c(
    option = "D",
    direction = 1,
    name = "Accuracy (%)",
    limits = c(0, 100),
    breaks = seq(0, 100, 20)
  ) +
  labs(
    title = "Per-Seed Final Test Accuracy",
    subtitle = "All configurations across experiments",
    x = NULL, y = NULL
  ) +
  theme_paper(base_size = 9) +
  theme(
    axis.text.x  = element_text(angle = 0, hjust = 0.5),
    axis.text.y  = element_text(size = rel(0.85)),
    panel.grid   = element_blank(),
    legend.position = "right"
  )

ggsave(file.path(fig_dir, "fig_heatmap_all_results.pdf"), p6,
       width = 7, height = 8, device = cairo_pdf)

# ==============================================================================
# FIGURE 7: Learning curves for Experiment 3 (Surrogate Gradients)
# ==============================================================================

cat("Generating Figure 7: Experiment 3 learning curves (surrogates)...\n")

exp3_curves <- bind_rows(
  build_curves_df(exp3_fast_sigmoid,     "Fast Sigmoid"),
  build_curves_df(exp3_arctan,           "Arctan"),
  build_curves_df(exp3_sigmoid,          "Sigmoid"),
  build_curves_df(exp3_triangular,       "Triangular"),
  build_curves_df(exp3_straight_through, "Straight-Through")
) %>%
  mutate(config = factor(config, levels = c("Fast Sigmoid", "Arctan", "Sigmoid",
                                            "Triangular", "Straight-Through")))

exp3_summary <- exp3_curves %>%
  group_by(config, epoch) %>%
  summarise(
    mean_acc = mean(test_acc_pct),
    sd_acc   = sd(test_acc_pct),
    .groups  = "drop"
  ) %>%
  mutate(
    lo = pmax(mean_acc - sd_acc, 0),
    hi = pmin(mean_acc + sd_acc, 100)
  )

# Map config names to surrogate colors for consistency with Figure 3
config_to_surr <- c(
  "Fast Sigmoid"    = "fast_sigmoid",
  "Arctan"          = "arctan",
  "Sigmoid"         = "sigmoid",
  "Triangular"      = "triangular",
  "Straight-Through" = "straight_through"
)

surr_colors_named <- setNames(surr_colors[config_to_surr], names(config_to_surr))

p7 <- ggplot() +
  # Individual seed traces
  geom_line(data = exp3_curves,
            aes(x = epoch, y = test_acc_pct, group = interaction(config, seed)),
            alpha = 0.25, linewidth = 0.4, color = "grey40") +
  # Mean +/- std ribbon
  geom_ribbon(data = exp3_summary,
              aes(x = epoch, ymin = lo, ymax = hi, fill = config),
              alpha = 0.15) +
  # Mean line
  geom_line(data = exp3_summary,
            aes(x = epoch, y = mean_acc, color = config),
            linewidth = 1) +
  facet_wrap(~ config, nrow = 1, scales = "free_y") +
  scale_color_manual(values = surr_colors_named) +
  scale_fill_manual(values = surr_colors_named) +
  labs(
    title    = "Experiment 3: Surrogate Gradient Comparison",
    subtitle = "Test accuracy over training epochs (mlx-snn on M3 Max)",
    x = "Epoch", y = "Test Accuracy (%)"
  ) +
  theme_paper(base_size = 9) +
  theme(legend.position = "none")

ggsave(file.path(fig_dir, "fig_learning_curves_exp3.pdf"), p7,
       width = 12, height = 3.5, device = cairo_pdf)

# ==============================================================================
# FIGURE 8: Recurrent weight initialization comparison (V configs)
# ==============================================================================

cat("Generating Figure 8: Recurrent V initialization comparison...\n")

# Collect final accuracy for each V config
recurrent_data <- bind_rows(
  collect_final(exp1_rleaky_orig,    "RLeaky",    "V = 1.0 (original)"),
  collect_final(exp1_rleaky,         "RLeaky",    "V = 0.5 (fixed)"),
  collect_final(exp1_rleaky_v01,     "RLeaky",    "V = 0.1 (learnable)"),
  collect_final(exp1_rsynaptic_orig, "RSynaptic", "V = 1.0 (original)"),
  collect_final(exp1_rsynaptic,      "RSynaptic", "V = 0.5 (fixed)"),
  collect_final(exp1_rsynaptic_v01,  "RSynaptic", "V = 0.1 (learnable)")
)

# Rename experiment column to V_config for this figure
recurrent_data <- recurrent_data %>%
  rename(neuron = config, v_config = experiment)

recurrent_summary <- recurrent_data %>%
  group_by(neuron, v_config) %>%
  summarise(
    mean_acc = mean(test_acc_pct),
    sd_acc   = sd(test_acc_pct),
    n_seeds  = n(),
    .groups  = "drop"
  )

# Order V configs logically
recurrent_summary <- recurrent_summary %>%
  mutate(v_config = factor(v_config, levels = c("V = 1.0 (original)",
                                                "V = 0.5 (fixed)",
                                                "V = 0.1 (learnable)")))

v_colors <- c(
  "V = 1.0 (original)" = "#E41A1C",
  "V = 0.5 (fixed)"    = "#377EB8",
  "V = 0.1 (learnable)" = "#4DAF4A"
)

p8 <- ggplot(recurrent_summary,
             aes(x = v_config, y = mean_acc, fill = v_config)) +
  geom_col(width = 0.6, color = "white", linewidth = 0.3) +
  geom_errorbar(aes(ymin = pmax(mean_acc - sd_acc, 0), ymax = mean_acc + sd_acc),
                width = 0.15, linewidth = 0.5) +
  # Individual seed points overlaid
  geom_point(data = recurrent_data %>%
               mutate(v_config = factor(v_config, levels = c("V = 1.0 (original)",
                                                             "V = 0.5 (fixed)",
                                                             "V = 0.1 (learnable)"))),
             aes(x = v_config, y = test_acc_pct),
             shape = 21, fill = "white", color = "black", size = 1.5,
             position = position_jitter(width = 0.1, seed = 42)) +
  facet_wrap(~ neuron, nrow = 1) +
  scale_fill_manual(values = v_colors, name = "V Init") +
  scale_y_continuous(limits = c(0, 105), breaks = seq(0, 100, 20)) +
  labs(
    title    = "Impact of Recurrent Weight Initialization (V)",
    subtitle = "Final test accuracy for RLeaky and RSynaptic neurons",
    x = NULL, y = "Test Accuracy (%)"
  ) +
  theme_paper() +
  theme(
    axis.text.x  = element_text(angle = 20, hjust = 1),
    legend.position = "none"
  )

ggsave(file.path(fig_dir, "fig_recurrent_V_comparison.pdf"), p8,
       width = 7, height = 4.5, device = cairo_pdf)

# ==============================================================================
# Summary
# ==============================================================================

cat("\n=======================================================\n")
cat("Figure generation complete.\n")
cat("=======================================================\n\n")

generated <- list.files(fig_dir, pattern = "\\.pdf$", full.names = TRUE)
cat(sprintf("Generated %d PDF figures in %s/:\n", length(generated), fig_dir))
for (f in generated) {
  info <- file.info(f)
  cat(sprintf("  %-45s  %s  (%.0f KB)\n",
              basename(f),
              format(info$mtime, "%Y-%m-%d %H:%M"),
              info$size / 1024))
}
cat("\nDone.\n")
