# GSMS 2020 Workshop:
#    Introduction to Bayesian statistics
#    Part 2 - Applications
# 
# Example 1: ANCOVA
# Paper:     Espinós, U., Fernandéz-Abascal, E.,G., & Ovejero, M.,(2019). Theory of
#            mind in remitted bipolar disorder: Interpersonal accuracy in 
#            recognition of dynamic nonverbal signals. PLoS ONE, 14(9), e0222112. 
#            doi: 10.1371/journal.pone.0222112.
# Data:      https://www.kaggle.com/mercheovejero/theory-of-mind-in-remitted-bipolar-disorder
# 
# Jorge N. Tendeiro, November 2020
# 



# 0. Prepare the R environment ----
library(magrittr)
library(knitr)
library(psych)
library(car)
library(heplots)
library(emmeans)
library(rstan)
options(mc.cores = parallel::detectCores() - 1)
rstan_options(auto_write = TRUE)
library(bayesplot)
library(gridExtra)
library(loo)
library(bridgesampling)
library(rstanarm)

setwd("slides/Part2-analysis/Example 1")



# 1. Import data ----
ToM.data <- read.csv("datasets_344408_680958_Database MiniPONS.csv", 
                     header = TRUE, sep = ";")
head(ToM.data)

# Variables:
#   Group:         Bipolar, Control, Depressive.
#   Type:          BD I, BD II, Control, Depressive.
#   Age:           In years
#   Right_answers: Number of right answers to the Mini PONS assessment.
# MiniPONS assessment scales:
#   Audio_prosody
#   Combined_channel
#   Face_video
#   Body_video
#   Positive_valence
#   Negative_valence
#   Dominant
#   Submissive
# A higher score in those scales means a better performance.

# Observe that:
# - Right_answers = Audio_prosody + Combined_channel + Face_video + Body_video:
all.equal(ToM.data[,"Right_answers"], rowSums(ToM.data[, 5:8]))
# - Right_answers = Positive_valence + Negative_valence:
all.equal(ToM.data[,"Right_answers"], rowSums(ToM.data[, 9:10]))
# - Right_answers = Dominant + Submissive:
all.equal(ToM.data[,"Right_answers"], rowSums(ToM.data[, 11:12]))

# We focus on one of the analysis only, see Table 3 in Espinós et al. (2019):
# Right_answers ~ Type + Age (and some variants for model comparison)
# This is an ANCOVA model, with Age as the covariate.
# Let's drop all unnecessary variables:
ToM.data <- ToM.data[, c("Type", "Age", "Right_answers")]
# Rename 'Type' (so 'Group' here corresponds to 'Type' in Espinós et al., 2019):
colnames(ToM.data)[1] <- "Group"
# Rename 'Right_answers':
colnames(ToM.data)[3] <- "y"
# Make sure Group is a factor:
ToM.data$Group <- factor(ToM.data$Group)



# 2. Descriptives ----
# These match with those in Table 2 in Espinós et al. (2019):
descrp.mat <- describeBy(ToM.data ~ Group, mat = TRUE, digits = 1)[-(1:4), 2:6]
row.names(descrp.mat) <- NULL
kable(descrp.mat[1:4, -2], format = 'latex', booktabs = TRUE) # Age
kable(descrp.mat[5:8, -2], format = 'latex', booktabs = TRUE) # Right_answers
# Clean once done:
rm(descrp.mat)

# Boxplot Type vs y:
png(filename = "../../include/figures/example1_boxplot.png", 
    width = 15, height = 10, units = "cm", pointsize = 10, res = 600)
par(mar = c(3.5, 3, 0, 0), mgp = c(2.5, 1, 0), bty = "n", bg = NA)
boxplot(ToM.data$y ~ ToM.data$Group, col = "#4b030680", bty = "n", 
        xlab = "Group", ylab = "y", ylim = c(25, 60), yaxt = "n", las = 1)
axis(2, seq(25, 60, 5), las = 1)
dev.off()
# Scatterplot Age vs y: 
png(filename = "../../include/figures/example1_scatterplot.png", 
    width = 15, height = 10, units = "cm", pointsize = 10, res = 600)
par(mar = c(3.5, 7, .5, .5), mgp = c(2.5, 1, 0), bty = "n", bg = NA)
plot(jitter(ToM.data$Age, 1), jitter(ToM.data$y, 1), pch = 4, col = "#4b0306", bty = "n", 
     xlab = "Age", ylab = "y", las = 1, 
     ylim = c(25, 60), yaxt = "n", yaxs = "i", 
     xlim = c(20, 80), xaxt = "n", xaxs = "i")
axis(1, seq(20, 80, 10), las = 1)
axis(2, seq(25, 60, 5), las = 1)
dev.off()



# 3. Frequentist analysis ----
ancova.freq <- lm(y ~ Group + Age, ToM.data)
summary(ancova.freq)
Anova(ancova.freq, type = "III")
etasq(ancova.freq, partial = TRUE)
ancova.freq %>% 
  emmeans(specs = "Group", by = "Age", 
                        contr = list(BDII.min.BDI  = c(-1, 1, 0, 0), 
                                     Ctrl.min.BDI  = c(-1, 0, 1, 0), 
                                     UD.min.BDI    = c(-1, 0, 0, 1), 
                                     BDII.min.Ctrl = c(0, 1, -1, 0), 
                                     BDII.min.UD   = c(0, 1, 0, -1), 
                                     Ctrl.min.UD   = c(0, 0, 1, -1))) %>% 
  confint()



# 4. Build Bayesian models ----
# I will entertain six different models; we can later compare:
# M1: y ~ 1                       <- baseline model (no effects)
# M2: y ~ Age                     <- SLR model
# M3: y ~ Group                   <- one-way ANOVA
# M4: y ~ Group + Age             <- ANCOVA, what I will focus on
# M5: y ~ Group + Age + Group:Age <- heterogeneous slopes model
# M6: M1 constrained such that mean(Control) = mean(UD), to assess this contrast.

# Models M1-M5 are all multiple linear regression models.
# The same Stan model works for all of them; all we need to do is to change the 
#   model matrix 'x' when calling the model.
# For model M1-M5, use MLR.stan.
# 
# M6 is different, as it requires a parameter constraint. 
# I coded this one in Stan separately; see MLR_contrast.stan.



# 5. Assess models through prior predictive checks ----
# For M4 only, as illustration.
x_priorPD <- model.matrix(y ~ Group + Age, ToM.data)
y_priorPD <- matrix(NA, 12, nrow(ToM.data))
set.seed(123)
for (i in 1:12)
{
  beta.i  <- rnorm(5, 0, 10)
  sigma.i <- abs(rcauchy(1, 0, 1))
  y_priorPD[i, ] <- rnorm(nrow(ToM.data), x_priorPD %*% beta.i, sigma.i)
}
# Plot:
png(filename = "../../include/figures/example1_priorPD.png", 
    width = 15, height = 10, units = "cm", pointsize = 10, res = 600)
par(mar = c(3.5, 3, 1, 0), mgp = c(2.5, 1, 0), bty = "n", bg = NA)
layout(matrix(1:12, 3, 4, byrow = TRUE))
# dens.y <- density(ToM.data$y)
# plot(dens.y, bty = "n", yaxt = "n", xlab = "", ylab = "", main = "Observed data")
# polygon(c(dens.y$x, rev(dens.y$x)), c(dens.y$y, rep(0, length(dens.y$x))), 
#         border = NA, col = "#4b0306")
for (i in 1:12)
{
  dens.i <- density(y_priorPD[i, ])
  plot(dens.i, bty = "n", yaxt = "n", xlab = "", ylab = "", main = paste0("Sim_", i))
  polygon(c(dens.i$x, rev(dens.i$x)), c(dens.i$y, rep(0, length(dens.i$x))), 
          border = NA, col = "#4b030640")
}
dev.off()
rm(x_priorPD, i, beta.i, sigma.i, dens.i)
rm(y_priorPD)

# Now with broader N(0, 100) prior for beta:
x_priorPD       <- model.matrix(y ~ Group + Age, ToM.data)
y_priorPD_broad <- matrix(NA, 12, nrow(ToM.data))
set.seed(456)
for (i in 1:12)
{
  beta.i  <- rnorm(5, 0, 100)
  sigma.i <- abs(rcauchy(1, 0, 1))
  y_priorPD_broad[i, ] <- rnorm(nrow(ToM.data), x_priorPD %*% beta.i, sigma.i)
}
# Plot:
png(filename = "../../include/figures/example1_priorPD_broad.png", 
    width = 15, height = 10, units = "cm", pointsize = 10, res = 600)
par(mar = c(3.5, 3, 1, 0), mgp = c(2.5, 1, 0), bty = "n", bg = NA)
layout(matrix(1:12, 3, 4, byrow = TRUE))
# dens.y <- density(ToM.data$y)
# plot(dens.y, bty = "n", yaxt = "n", xlab = "", ylab = "", main = "Observed data")
# polygon(c(dens.y$x, rev(dens.y$x)), c(dens.y$y, rep(0, length(dens.y$x))), 
#         border = NA, col = "#4b0306")
for (i in 1:12)
{
  dens.i <- density(y_priorPD_broad[i, ])
  plot(dens.i, bty = "n", yaxt = "n", xlab = "", ylab = "", main = paste0("Sim_", i))
  polygon(c(dens.i$x, rev(dens.i$x)), c(dens.i$y, rep(0, length(dens.i$x))), 
          border = NA, col = "#4b030640")
}
dev.off()
rm(x_priorPD, i, beta.i, sigma.i, dens.i)
rm(y_priorPD_broad)

# Now with shrink N(0, .1) prior for beta:
x_priorPD        <- model.matrix(y ~ Group + Age, ToM.data)
y_priorPD_shrink <- matrix(NA, 12, nrow(ToM.data))
set.seed(789)
for (i in 1:12)
{
  beta.i  <- rnorm(5, 0, .1)
  sigma.i <- abs(rcauchy(1, 0, 1))
  y_priorPD_shrink[i, ] <- rnorm(nrow(ToM.data), x_priorPD %*% beta.i, sigma.i)
}
# Plot:
png(filename = "../../include/figures/example1_priorPD_shrink.png", 
    width = 15, height = 10, units = "cm", pointsize = 10, res = 600)
par(mar = c(3.5, 3, 1, 0), mgp = c(2.5, 1, 0), bty = "n", bg = NA)
layout(matrix(1:12, 3, 4, byrow = TRUE))
# dens.y <- density(ToM.data$y)
# plot(dens.y, bty = "n", yaxt = "n", xlab = "", ylab = "", main = "Observed data")
# polygon(c(dens.y$x, rev(dens.y$x)), c(dens.y$y, rep(0, length(dens.y$x))), 
#         border = NA, col = "#4b0306")
for (i in 1:12)
{
  dens.i <- density(y_priorPD_shrink[i, ])
  plot(dens.i, bty = "n", yaxt = "n", xlab = "", ylab = "", main = paste0("Sim_", i))
  polygon(c(dens.i$x, rev(dens.i$x)), c(dens.i$y, rep(0, length(dens.i$x))), 
          border = NA, col = "#4b030640")
}
dev.off()
rm(x_priorPD, i, beta.i, sigma.i, dens.i)
rm(y_priorPD_shrink)



# 6. Fit Bayesian models ----
#    6.1 Model M1 ----
x.M1    <- model.matrix(y ~ 1, ToM.data)
data.M1 <- list(N = nrow(ToM.data), 
                K = ncol(x.M1) - 1, 
                x = x.M1, 
                y = ToM.data$y)
fit.M1  <- stan("MLR.stan", 
                data   = data.M1, 
                pars   = c("beta", "sigma", "R2", "y_ppd", "log_lik"), 
                chains = 3)
summary(fit.M1, pars = c("beta", "sigma", "R2"), 
        probs = c(.025, .5, .975))$summary %>% round(3)
rm(x.M1, data.M1)

#    6.2 Model M2 ----
x.M2    <- model.matrix(y ~ Age, ToM.data)
data.M2 <- list(N = nrow(ToM.data), 
                K = ncol(x.M2) - 1, 
                x = x.M2, 
                y = ToM.data$y)
fit.M2  <- stan("MLR.stan", 
                data   = data.M2, 
                pars   = c("beta", "sigma", "R2", "y_ppd", "log_lik"), 
                chains = 3)
summary(fit.M2, pars = c("beta", "sigma", "R2"), 
        probs = c(.025, .5, .975))$summary %>% round(3)
rm(x.M2, data.M2)

#    6.3 Model M3 ----
x.M3    <- model.matrix(y ~ Group, ToM.data)
data.M3 <- list(N = nrow(ToM.data), 
                K = ncol(x.M3) - 1, 
                x = x.M3, 
                y = ToM.data$y)
fit.M3  <- stan("MLR.stan", 
                data   = data.M3, 
                pars   = c("beta", "sigma", "R2", "y_ppd", "log_lik"), 
                chains = 3)
summary(fit.M3, pars = c("beta", "sigma", "R2"), 
        probs = c(.025, .5, .975))$summary %>% round(3)
rm(x.M3, data.M3)

#    6.4 Model M4 ----
x.M4    <- model.matrix(y ~ Group + Age, ToM.data)
data.M4 <- list(N = nrow(ToM.data), 
                K = ncol(x.M4) - 1, 
                x = x.M4, 
                y = ToM.data$y)
fit.M4  <- stan("MLR.stan", 
                data   = data.M4, 
                pars   = c("beta", "sigma", "R2", "y_ppd", "log_lik"), 
                chains = 3)
summary(fit.M4, pars = c("beta", "sigma", "R2"), 
        probs = c(.025, .5, .975))$summary %>% round(3)
rm(x.M4, data.M4)

#    6.5 Model M5 ----
x.M5    <- model.matrix(y ~ Group * Age, ToM.data)
data.M5 <- list(N = nrow(ToM.data), 
                K = ncol(x.M5) - 1, 
                x = x.M5, 
                y = ToM.data$y)
fit.M5  <- stan("MLR.stan", 
                data   = data.M5, 
                pars   = c("beta", "sigma", "R2", "y_ppd", "log_lik"), 
                chains = 3)
summary(fit.M5, pars = c("beta", "sigma", "R2"), 
        probs = c(.025, .5, .975))$summary %>% round(3)
rm(x.M5, data.M5)

#    6.6 Model M6 ----
x.M6    <- x.M4
data.M6 <- data.M4
fit.M6  <- stan("MLR_contrast.stan", 
                data   = data.M6, 
                pars   = c("beta", "sigma", "R2", "y_ppd", "log_lik"), 
                chains = 3)
summary(fit.M6, pars = c("beta", "sigma", "R2"), 
        probs = c(.025, .5, .975))$summary %>% round(3)
rm(x.M6, data.M6)



# 7. MCMC diagnostics ----
#    7.1 Model M4 ----

# Trace plot:
color_scheme_set("red")
mcmc_trace(as.array(fit.M4), pars = vars(contains("beta"), "sigma"))
# The chains mixed well.
ggsave("../../include/figures/example1_M4_traceplot.png",
  plot   = last_plot(),
  device = "png",
  dpi    = 300, 
  bg     = "transparent",
  width = 20, height = 10, unit = "cm"
)
dev.off()

# Rhat:
mcmc_rhat(rhat(fit.M4, pars = c("beta", "sigma"))) + yaxis_text(hjust = 1)
# All below 1.05, typically good news.
ggsave("../../include/figures/example1_M4_Rhat.png",
       plot   = last_plot(),
       device = "png",
       dpi    = 300,
       bg     = "transparent",
       width = 20, height = 10, unit = "cm"
)
dev.off()

# Effective sample size:
mcmc_neff(neff_ratio(fit.M4, pars = c("beta", "sigma")), size = 2) + yaxis_text(hjust = 1)
# All above 0.1, typically good news.
ggsave("../../include/figures/example1_M4_Neff.png",
       plot   = last_plot(),
       device = "png",
       dpi    = 300,
       bg     = "transparent",
       width = 20, height = 10, unit = "cm"
)
dev.off()

# Auto-correlation:
mcmc_acf(fit.M4, pars = vars(contains("beta"), "sigma"), lags = 10)
# All seem low enough.
ggsave("../../include/figures/example1_M4_autocorr.png",
       plot   = last_plot(),
       device = "png",
       dpi    = 300,
       bg     = "transparent",
       width = 20, height = 10, unit = "cm"
)
dev.off()



# 8. Assess model fit through posterior predictive checks ----
#    8.1 Model M4 ----
y.ppd.M4 <- as.matrix(fit.M4, pars = "y_ppd")
color_scheme_set("red")

# Distribution of y:
ppc_dens_overlay(ToM.data$y, y.ppd.M4[1:100, ])+ 
  theme(axis.line.y=element_blank())
ggsave("../../include/figures/example1_M4_posteriorPD_y.png",
       plot   = last_plot(),
       device = "png",
       dpi    = 300, 
       bg     = "transparent",
       width  = 20, height = 15, unit = "cm"
)
dev.off()

# Distribution of y per group:
g1 <- which(ToM.data$Group == "BD I")
p1 <- ppc_dens_overlay(ToM.data$y[g1], y.ppd.M4[1:100, g1]) + 
  theme(axis.line.y=element_blank())  + ggtitle("BD I")
g2 <- which(ToM.data$Group == "BD II")
p2 <- ppc_dens_overlay(ToM.data$y[g2], y.ppd.M4[1:100, g2]) + 
  theme(axis.line.y=element_blank()) + ggtitle("BD II")
g3 <- which(ToM.data$Group == "Control")
p3 <- ppc_dens_overlay(ToM.data$y[g3], y.ppd.M4[1:100, g3]) + 
  theme(axis.line.y=element_blank()) + ggtitle("Control")
g4 <- which(ToM.data$Group == "UD")
p4 <- ppc_dens_overlay(ToM.data$y[g4], y.ppd.M4[1:100, g4]) + 
  theme(axis.line.y=element_blank()) + ggtitle("UD")
ggsave("../../include/figures/example1_M4_posteriorPD_y_group.png",
       plot   = grid.arrange(p1, p2, p3, p4, ncol = 2),
       device = "png",
       dpi    = 300, 
       bg     = "transparent",
       width  = 20, height = 15, unit = "cm"
)
rm(g1, p1, g2, p2, g3, p3, g4, p4)

# Various statistics of y:
p1 <- ppc_stat(y = ToM.data$y, yrep = y.ppd.M4, stat = "mean") + 
  theme(axis.line.y=element_blank())
p2 <- ppc_stat(y = ToM.data$y, yrep = y.ppd.M4, stat = "sd")+ 
  theme(axis.line.y=element_blank())
cor.y_Age <- function(vec) cor(vec, ToM.data$Age)
p3 <- ppc_stat(y = ToM.data$y, yrep = y.ppd.M4, stat = cor.y_Age)+ 
  theme(axis.line.y=element_blank())
p4 <- ppc_stat(y = ToM.data$y, yrep = y.ppd.M4, stat = "max")+ 
  theme(axis.line.y=element_blank())
ggsave("../../include/figures/example1_M4_posteriorPD_stats.png",
       plot   = grid.arrange(p1, p2, p3, p4, ncol = 2),
       device = "png",
       dpi    = 300, 
       bg     = "transparent",
       width  = 20, height = 15, unit = "cm"
)
rm(p1, p2, p3, p4, cor.y_Age)
rm(y.ppd.M4)

#    8.2 Model M1 ----
y.ppd.M1 <- as.matrix(fit.M1, pars = "y_ppd")
color_scheme_set("red")

# Distribution of y:
ppc_dens_overlay(ToM.data$y, y.ppd.M1[1:100, ])+ 
  theme(axis.line.y=element_blank())
ggsave("../../include/figures/example1_M1_posteriorPD_y.png",
       plot   = last_plot(),
       device = "png",
       dpi    = 300, 
       bg     = "transparent",
       width  = 20, height = 15, unit = "cm"
)
dev.off()

# Distribution of y per group:
g1 <- which(ToM.data$Group == "BD I")
p1 <- ppc_dens_overlay(ToM.data$y[g1], y.ppd.M1[1:100, g1]) + 
  theme(axis.line.y=element_blank())  + ggtitle("BD I")
g2 <- which(ToM.data$Group == "BD II")
p2 <- ppc_dens_overlay(ToM.data$y[g2], y.ppd.M1[1:100, g2]) + 
  theme(axis.line.y=element_blank()) + ggtitle("BD II")
g3 <- which(ToM.data$Group == "Control")
p3 <- ppc_dens_overlay(ToM.data$y[g3], y.ppd.M1[1:100, g3]) + 
  theme(axis.line.y=element_blank()) + ggtitle("Control")
g4 <- which(ToM.data$Group == "UD")
p4 <- ppc_dens_overlay(ToM.data$y[g4], y.ppd.M1[1:100, g4]) + 
  theme(axis.line.y=element_blank()) + ggtitle("UD")
ggsave("../../include/figures/example1_M1_posteriorPD_y_group.png",
       plot   = grid.arrange(p1, p2, p3, p4, ncol = 2),
       device = "png",
       dpi    = 300, 
       bg     = "transparent",
       width  = 20, height = 15, unit = "cm"
)
rm(g1, p1, g2, p2, g3, p3, g4, p4)

# Various statistics of y:
p1 <- ppc_stat(y = ToM.data$y, yrep = y.ppd.M1, stat = "mean") + 
  theme(axis.line.y=element_blank())
p2 <- ppc_stat(y = ToM.data$y, yrep = y.ppd.M1, stat = "sd")+ 
  theme(axis.line.y=element_blank())
cor.y_Age <- function(vec) cor(vec, ToM.data$Age)
p3 <- ppc_stat(y = ToM.data$y, yrep = y.ppd.M1, stat = cor.y_Age)+ 
  theme(axis.line.y=element_blank())
p4 <- ppc_stat(y = ToM.data$y, yrep = y.ppd.M1, stat = "max")+ 
  theme(axis.line.y=element_blank())
ggsave("../../include/figures/example1_M1_posteriorPD_stats.png",
       plot   = grid.arrange(p1, p2, p3, p4, ncol = 2),
       device = "png",
       dpi    = 300, 
       bg     = "transparent",
       width  = 20, height = 15, unit = "cm"
)
rm(p1, p2, p3, p4, cor.y_Age)
rm(y.ppd.M1)



# 9. Model comparison ----
#    9.1 LOO-CV ----
#        Step 1: Extract pointwise log-likelihood.
log.lik.M1  <- extract_log_lik(fit.M1, merge_chains = FALSE)
log.lik.M2  <- extract_log_lik(fit.M2, merge_chains = FALSE)
log.lik.M3  <- extract_log_lik(fit.M3, merge_chains = FALSE)
log.lik.M4  <- extract_log_lik(fit.M4, merge_chains = FALSE)
log.lik.M5  <- extract_log_lik(fit.M5, merge_chains = FALSE)
log.lik.M6  <- extract_log_lik(fit.M6, merge_chains = FALSE)

#        Step 2: Compute relative effective sample sizes.
r.eff.M1  <- relative_eff(exp(log.lik.M1),  cores = 3)
r.eff.M2  <- relative_eff(exp(log.lik.M2),  cores = 3)
r.eff.M3  <- relative_eff(exp(log.lik.M3),  cores = 3)
r.eff.M4  <- relative_eff(exp(log.lik.M4),  cores = 3)
r.eff.M5  <- relative_eff(exp(log.lik.M5),  cores = 3)
r.eff.M6  <- relative_eff(exp(log.lik.M6),  cores = 3)

#        Step 3: Compute the PSIS-LOO CV, which is an efficient approximation 
#                of leave-one-out (LOO) cross-validation for Bayesian models, 
#                using Pareto smoothed importance sampling (PSIS).
loo.M1  <- loo(log.lik.M1, r_eff = r.eff.M1, cores = 3)
loo.M2  <- loo(log.lik.M2, r_eff = r.eff.M2, cores = 3)
loo.M3  <- loo(log.lik.M3, r_eff = r.eff.M3, cores = 3)
loo.M4  <- loo(log.lik.M4, r_eff = r.eff.M4, cores = 3)
loo.M5  <- loo(log.lik.M5, r_eff = r.eff.M5, cores = 3)
loo.M6  <- loo(log.lik.M6, r_eff = r.eff.M6, cores = 3)

#        Step 4: Finally, compare the models.
loo_compare(loo.M1, loo.M2, loo.M3, loo.M4, loo.M5, loo.M6)[, c(1:2, 7)] %>% 
  kable(format = 'latex', booktabs = TRUE, digits = 1)
# elpd_diff = estimated difference of expected LOO prediction errors between 
#             the models
# Thus, the models (=rows) are order in decreasing predictive accuracy, 
#    from the best (M4) to the worst (M1).
# Aki Vehtari advices that the ELPD of one model is at least 4xSE lower than 
#    that of another model for the latter to be clearly favoured:
# https://discourse.mc-stan.org/t/interpreting-output-from-compare-of-loo/3380/2). 
# Using this rule, model M2 and M1 clearly predict worse than M4. But M5, M3, and
#    M6 shouldn't yet be discarded based on this criterion alone.

# Clean:
rm(log.lik.M1, log.lik.M2, log.lik.M3, log.lik.M4, log.lik.M5, log.lik.M6, 
   r.eff.M1, r.eff.M2, r.eff.M3, r.eff.M4, r.eff.M5, r.eff.M6, 
   loo.M1, loo.M2, loo.M3, loo.M4, loo.M5, loo.M6)

#    9.2 Bayes factors (bridge sampling) ----
# For this, we must re-fit the Stan models, but this time with plenty more samples.
# See vignette("bridgesampling_example_stan")
#        9.2.1 Model M1 ----
x.M1    <- model.matrix(y ~ 1, ToM.data)
data.M1 <- list(N = nrow(ToM.data), 
                K = ncol(x.M1) - 1, 
                x = x.M1, 
                y = ToM.data$y)
fit.M1.bs  <- stan(fit    = fit.M1, 
                   data   = data.M1, 
                   pars   = c("beta", "sigma"), 
                   iter   = 51000, 
                   warmup = 1000,
                   chains = 3)
# Compute the log marginal likelihood:
M1.bridge <- bridge_sampler(fit.M1.bs, silent = TRUE)

#        9.2.2 Model M2 ----
x.M2    <- model.matrix(y ~ Age, ToM.data)
data.M2 <- list(N = nrow(ToM.data), 
                K = ncol(x.M2) - 1, 
                x = x.M2, 
                y = ToM.data$y)
fit.M2.bs  <- stan(fit    = fit.M2, 
                   data   = data.M2, 
                   pars   = c("beta", "sigma"), 
                   iter   = 51000, 
                   warmup = 1000,
                   chains = 3)
# Compute the log marginal likelihood:
M2.bridge <- bridge_sampler(fit.M2.bs, silent = TRUE)

#        9.2.3 Model M3 ----
x.M3    <- model.matrix(y ~ Group, ToM.data)
data.M3 <- list(N = nrow(ToM.data), 
                K = ncol(x.M3) - 1, 
                x = x.M3, 
                y = ToM.data$y)
fit.M3.bs  <- stan(fit    = fit.M3, 
                   data   = data.M3, 
                   pars   = c("beta", "sigma"), 
                   iter   = 51000, 
                   warmup = 1000,
                   chains = 3)
# Compute the log marginal likelihood:
M3.bridge <- bridge_sampler(fit.M3.bs, silent = TRUE)

#        9.2.4 Model M4 ----
x.M4    <- model.matrix(y ~ Group + Age, ToM.data)
data.M4 <- list(N = nrow(ToM.data), 
                K = ncol(x.M4) - 1, 
                x = x.M4, 
                y = ToM.data$y)
fit.M4.bs  <- stan(fit    = fit.M4, 
                   data   = data.M4, 
                   pars   = c("beta", "sigma"), 
                   iter   = 51000, 
                   warmup = 1000,
                   chains = 3)
# Compute the log marginal likelihood:
M4.bridge <- bridge_sampler(fit.M4.bs, silent = TRUE)

#        9.2.5 Model M5 ----
x.M5    <- model.matrix(y ~ Group * Age, ToM.data)
data.M5 <- list(N = nrow(ToM.data), 
                K = ncol(x.M5) - 1, 
                x = x.M5, 
                y = ToM.data$y)
fit.M5.bs  <- stan(fit    = fit.M5, 
                   data   = data.M5, 
                   pars   = c("beta", "sigma"), 
                   iter   = 51000, 
                   warmup = 1000,
                   chains = 3)
# Compute the log marginal likelihood:
M5.bridge <- bridge_sampler(fit.M5.bs, silent = TRUE)

#        9.2.6 Model M6 ----
x.M6    <- x.M4
data.M6 <- data.M4
fit.M6.bs  <- stan(fit    = fit.M6, 
                   data   = data.M6, 
                   pars   = c("beta0", "sigma"), 
                   iter   = 51000, 
                   warmup = 1000,
                   chains = 3)
# Compute the log marginal likelihood:
M6.bridge <- bridge_sampler(fit.M6.bs, silent = TRUE)

# Clean up:
rm(x.M1, x.M2, x.M3, x.M4, x.M5, x.M6, 
   data.M1, data.M2, data.M3, data.M4, data.M5, data.M6)

#        9.2.7 Compute Bayes factors ----
bf(M2.bridge, M1.bridge)
bf(M3.bridge, M1.bridge)
bf(M4.bridge, M1.bridge)
bf(M5.bridge, M1.bridge)
bf(M6.bridge, M1.bridge)
bf(M3.bridge, M2.bridge)
bf(M4.bridge, M2.bridge)
bf(M5.bridge, M2.bridge)
bf(M6.bridge, M2.bridge)
bf(M4.bridge, M3.bridge)
bf(M3.bridge, M5.bridge)
bf(M3.bridge, M6.bridge)
bf(M4.bridge, M5.bridge)
bf(M4.bridge, M6.bridge)
bf(M5.bridge, M6.bridge)
# I find these results just horrible. 
# E.g., LOO indicated that the ANCOVA model and the heterogeneous slopes 
#    model have quite similar out-of-sample predictive abilities.
# Now the corresponding BF favours ANCOVA by over 2 million times (loool).
# I doubted my own computations, so I tried to reproduce them through the 
#    rstanarm package. See below.

# M4:
fit.M4.bs.rsa <- stan_glm(y ~ Group + Age, data = ToM.data,
                          chains = 3, iter = 51000, warmup = 1000, 
                          diagnostic_file = file.path(tempdir(), "df4.csv"), 
                          prior = normal(0, 10, autoscale = FALSE),
                          prior_intercept = normal(0, 10, autoscale = FALSE),
                          prior_aux = cauchy(0, 1, autoscale = FALSE)
)
M4.bridge.rsa <- bridge_sampler(fit.M4.bs.rsa)

# M5:
fit.M5.bs.rsa <- stan_glm(y ~ Group * Age, data = ToM.data,
                          chains = 3, iter = 51000, warmup = 1000, 
                          diagnostic_file = file.path(tempdir(), "df5.csv"), 
                          prior = normal(0, 10, autoscale = FALSE),
                          prior_intercept = normal(0, 10, autoscale = FALSE),
                          prior_aux = cauchy(0, 1, autoscale = FALSE)
)
M5.bridge.rsa <- bridge_sampler(fit.M5.bs.rsa)

bf(M4.bridge.rsa, M5.bridge.rsa) # similar to mine

# Furthermore, changing the priors matters a lot.
# Here's the results when we allow rstanarm to autoscale the priors:
# M4:
fit.M4.bs.rsa2 <- stan_glm(y ~ Group + Age, data = ToM.data,
                          chains = 3, iter = 51000, warmup = 1000, 
                          diagnostic_file = file.path(tempdir(), "df4.csv"), 
                          prior = normal(0, 10, autoscale = TRUE),
                          prior_intercept = normal(0, 10, autoscale = TRUE),
                          prior_aux = cauchy(0, 1, autoscale = TRUE)
)
M4.bridge.rsa2 <- bridge_sampler(fit.M4.bs.rsa2)

# M5:
fit.M5.bs.rsa2 <- stan_glm(y ~ Group * Age, data = ToM.data,
                          chains = 3, iter = 51000, warmup = 1000, 
                          diagnostic_file = file.path(tempdir(), "df5.csv"), 
                          prior = normal(0, 10, autoscale = TRUE),
                          prior_intercept = normal(0, 10, autoscale = TRUE),
                          prior_aux = cauchy(0, 1, autoscale = TRUE)
)
M5.bridge.rsa2 <- bridge_sampler(fit.M5.bs.rsa2)

bf(M4.bridge.rsa2, M5.bridge.rsa2) # 24709.3, so 2 orders of magnitude smaller

# And here's the results when we use rstanarm's default priors:
# M4:
fit.M4.bs.rsa3 <- stan_glm(y ~ Group + Age, data = ToM.data,
                           chains = 3, iter = 51000, warmup = 1000, 
                           diagnostic_file = file.path(tempdir(), "df4.csv")#, 
                           # prior = normal(0, 10, autoscale = TRUE),
                           # prior_intercept = normal(0, 10, autoscale = TRUE),
                           # prior_aux = cauchy(0, 1, autoscale = TRUE)
)
M4.bridge.rsa3 <- bridge_sampler(fit.M4.bs.rsa3)

# M5:
fit.M5.bs.rsa3 <- stan_glm(y ~ Group * Age, data = ToM.data,
                           chains = 3, iter = 51000, warmup = 1000, 
                           diagnostic_file = file.path(tempdir(), "df5.csv")#, 
                           # prior = normal(0, 10, autoscale = TRUE),
                           # prior_intercept = normal(0, 10, autoscale = TRUE),
                           # prior_aux = cauchy(0, 1, autoscale = TRUE)
)
M5.bridge.rsa3 <- bridge_sampler(fit.M5.bs.rsa3)

bf(M4.bridge.rsa3, M5.bridge.rsa3) # 401.8, so 4 orders of magnitude smaller

# Finally, I also used JASP 0.13.1 with its default priors: BF = 54.806

# I decided to drop BFs from the presentation.

# Clean:
rm(fit.M1, fit.M2, fit.M3, fit.M5, fit.M6, 
   fit.M1.bs, fit.M2.bs, fit.M3.bs, fit.M4.bs, fit.M5.bs, fit.M6.bs, 
   M1.bridge, M2.bridge, M3.bridge, M4.bridge, M5.bridge, M6.bridge, 
   fit.M4.bs.rsa, fit.M4.bs.rsa2, fit.M4.bs.rsa3, 
   fit.M5.bs.rsa, fit.M5.bs.rsa2, fit.M5.bs.rsa3, 
   M4.bridge.rsa, M4.bridge.rsa2, M4.bridge.rsa3, 
   M5.bridge.rsa, M5.bridge.rsa2, M5.bridge.rsa3)



# 10. Summarize results (ANCOVA only) ----
#     To run this section, we only need to previously run sections: 
#        0, 1, 3, and 6.4.

#     10.1 ANCOVA plots per group ----
bayes.coef <- summary(fit.M4, pars = c("beta", "sigma"))$summary[, "50%"]
bayes.int <- c(bayes.coef[1], bayes.coef[1] + bayes.coef[2:4])
bayes.slope <- bayes.coef[5]
# 
bayes.draws <- as.matrix(fit.M4, pars = c("beta"))
bayes.int.draws <- cbind(bayes.draws[, 1], bayes.draws[, 1] + bayes.draws[, 2:4])
bayes.slope.draws <- bayes.draws[, 5]

png(filename = "../../include/figures/example1_M4_post_groups.png", 
    width = 15, height = 10, units = "cm", pointsize = 10, res = 600)
# par(mar = c(3.5, 3, 0, 0), mgp = c(2.5, 1, 0), bty = "n", bg = NA)

par(mar = c(3.5, 3.5, 1, .5), bg = NA)
layout(matrix(1:4, 2, 2, byrow = TRUE))
for (i in 1:4) {
  data1.tmp <- ToM.data[ToM.data$Group == levels(factor(ToM.data$Group))[i], ]
  plot(jitter(data1.tmp$Age, 1), data1.tmp$y, 
       xlim = c(20, 80), ylim = c(25, 65), xaxt = "n", yaxt = "n", 
       pch = 21, bg = "red", bty = "n", xlab = "", ylab = "", xaxs = "i", yaxs = "i", 
       main = levels(factor(ToM.data$Group))[i])
  axis(1, seq(20, 80, 20))
  if (i > 2) mtext("Age", 1, 2)
  axis(2, seq(25, 65, 10), las = 1)
  if (i %% 2 == 1) mtext("y", 2, 2.5)
  for (rep in 1:200)
  {
    abline(bayes.int.draws[rep, i], bayes.slope.draws[rep], col =  "gray", lwd = .4)
  }
  abline(bayes.int[i], bayes.slope, col = 2, lwd = 4)
}
dev.off()

rm(bayes.coef, bayes.int, bayes.slope, 
   bayes.draws, bayes.int.draws, bayes.slope.draws, 
   i, rep, data1.tmp)

#     10.2 Contrast Control vs UD ----
est.M4             <- as.matrix(fit.M4)[, 1:7]
Control.min.UD     <- est.M4[, "beta[3]"] - est.M4[, "beta[4]"]
Control.min.UD.qts <- quantile(Control.min.UD, probs = c(.025, .50, .975)) %>% round(1)
prob.L0            <- mean(Control.min.UD > 0)

png(filename = "../../include/figures/example1_M4_post_ControlminUD.png", 
    width = 15, height = 10, units = "cm", pointsize = 10, res = 600)
par(mar = c(3.5, 3, 1, 0), mgp = c(2.5, 1, 0), bty = "n", bg = NA)
dens <- density(Control.min.UD, from = 2, to = 9)
plot(dens, bty = "n", yaxt = "n", xlab = "", ylab = "", main = paste0("Control - UD"), 
     xlim = c(2, 9))
polygon(c(dens$x, rev(dens$x)), c(dens$y, rep(0, length(dens$x))), 
        border = NA, col = "#4b030640")
text(3, .3, paste0("Median = ", Control.min.UD.qts[2]))
text(3, .27, paste0("95% CI = (", Control.min.UD.qts[1], ", ", Control.min.UD.qts[3], ")"))
text(3, .24, paste0("Prob(Control > UD) = ", prob.L0))
dev.off()
rm(Control.min.UD, Control.min.UD.qts, prob.L0, dens)

#     10.3 All pairwise contrasts ----
contrasts.mat <- cbind(est.M4[, "beta[2]"], 
                       est.M4[, "beta[3]"], 
                       est.M4[, "beta[4]"], 
                       est.M4[, "beta[2]"] - est.M4[, "beta[3]"], 
                       est.M4[, "beta[2]"] - est.M4[, "beta[4]"], 
                       est.M4[, "beta[3]"] - est.M4[, "beta[4]"])
contrasts.names <- c("BD II - BD I", 
                     "Control - BD I", 
                     "UD - BD I", 
                     "BD II - Control", 
                     "BD II - UD", 
                     "Control - UD")
contrasts.L0 <- c("P(> 0) = ", 
                  "P(> 0) = ", 
                  "P(> 0) = ", 
                  "P(> 0) = ", 
                  "P(> 0) = ", 
                  "P(> 0) = ")
png(filename = "../../include/figures/example1_M4_post_pairwise_contrasts.png", 
    width = 15, height = 10, units = "cm", pointsize = 10, res = 600)
par(mar = c(3.5, 3, 1, 0), mgp = c(2.5, 1, 0), bty = "n", bg = NA)
layout(matrix(1:6, 2, 3, byrow = TRUE))
for (i in 1:6)
{
  dens.i <- density(contrasts.mat[, i], n = 1001)
  plot(dens.i, bty = "n", yaxt = "n", xlab = "", ylab = "", main = contrasts.names[i])
  polygon(c(dens.i$x, rev(dens.i$x)), c(dens.i$y, rep(0, length(dens.i$x))), 
          border = NA, col = "#4b030640")
  qts     <- quantile(contrasts.mat[, i], probs = c(.025, .50, .975)) %>% round(1)
  prob.L0 <- mean(contrasts.mat[, i] > 0) %>% round(1)
  CI.low  <- which.min(abs(dens.i$x - qts[1]))
  CI.upp  <- which.min(abs(dens.i$x - qts[3]))
  polygon(c(dens.i$x[CI.low:CI.upp], rev(dens.i$x[CI.low:CI.upp])), 
          c(dens.i$y[CI.low:CI.upp], rep(0, length(dens.i$x[CI.low:CI.upp]))), 
          border = NA, col = "#4b030640")
  abline(v = c(dens.i$x[CI.low], dens.i$x[CI.upp]), col = "#4b0306", lwd = .5, lty = 2)
  abline(v = qts[2], lwd = 2, col = "#4b0306")
  text(dens.i$x[850], .9*dens.i$y[which.min(abs(dens.i$x - qts[2]))], 
       paste0(contrasts.L0[i], prob.L0))
  rm(dens.i, qts, prob.L0)
}
dev.off()
rm(contrasts.mat, contrasts.names, contrasts.L0, i, CI.low, CI.upp)

#     10.4 Prediction when Age = 70, Group = UD ----
mean.UD.70     <- est.M4[, 1:5] %*% c(1, 0, 0, 1, 70)
pred.UD.70     <- rnorm(3000, mean.UD.70, est.M4[, "sigma"])
pred.UD.70.qts <- quantile(pred.UD.70, probs = c(.025, .50, .975)) %>% round(1)

png(filename = "../../include/figures/example1_M4_post_Pred_UD70.png", 
    width = 15, height = 10, units = "cm", pointsize = 10, res = 600)
par(mar = c(3.5, 3, 1, 0), mgp = c(2.5, 1, 0), bty = "n", bg = NA)
dens <- density(pred.UD.70, from = 20, to = 60)
plot(dens, bty = "n", yaxt = "n", xlab = "", ylab = "", 
     main = paste0("Group = UD, Age = 70"), 
     xlim = c(20, 60))
polygon(c(dens$x, rev(dens$x)), c(dens$y, rep(0, length(dens$x))), 
        border = NA, col = "#4b030640")
CI.low  <- which.min(abs(dens$x - pred.UD.70.qts[1]))
CI.upp  <- which.min(abs(dens$x - pred.UD.70.qts[3]))
polygon(c(dens$x[CI.low:CI.upp], rev(dens$x[CI.low:CI.upp])), 
        c(dens$y[CI.low:CI.upp], rep(0, length(dens$x[CI.low:CI.upp]))), 
        border = NA, col = "#4b030640")
abline(v = c(dens$x[CI.low], dens$x[CI.upp]), col = "#4b0306", lwd = .5, lty = 2)
abline(v = pred.UD.70.qts[2], lwd = 2, col = "#4b0306")
text(25, .07, paste0("Median = ", pred.UD.70.qts[2]))
text(25, .06, paste0("95% CI = (", pred.UD.70.qts[1], ", ", pred.UD.70.qts[3], ")"))
dev.off()
rm(mean.UD.70, pred.UD.70, pred.UD.70.qts, dens, CI.low, CI.upp)

#     10.5 Posterior distributions of parameters ----
stan_dens(fit.M4, pars = c("sigma", "R2"))
ggsave("../../include/figures/example1_M4_post_pars.png",
       plot   = last_plot(),
       device = "png",
       dpi    = 300, 
       bg     = "transparent",
       width = 15, height = 10, unit = "cm"
)
dev.off()

#     10.5 Numeric summaries ----
post.summ <- est.M4[, 1:7]
cbind("Mean"  = colMeans(post.summ), 
      "SD"    = apply(post.summ, 2, sd), 
      "2.5%"  = apply(post.summ, 2, function(vec) quantile(vec, .025)), 
      "97.5%" =  apply(post.summ, 2, function(vec) quantile(vec, .975))) %>% 
  round(2) %>% 
  kable(format = 'latex', booktabs = TRUE) 
