library(ggplot2)
library(ggthemes)
setwd("/Users/danburrows/Desktop/multilevel_bootstrap_particle_filter_swe/steady_state_ode/1d/final")

rm(list=ls())
set.seed(13)

# Open the files for reading
hmm_data = file("hmm_data.txt", "r")
raw_bpf_mse = file("raw_bpf_mse.txt", "r")
raw_bpf_ks = file("raw_bpf_ks.txt", "r")
ml_parameters = file("ml_parameters.txt", "r")
N1s_data = file("N1s_data.txt", "r")
raw_mse = file("raw_mse.txt", "r")
raw_ks = file("raw_ks.txt", "r")

# HMM data
data_length = strtoi(readLines(hmm_data, n=1))
line = strsplit(readLines(hmm_data, n=1), " ")
sig_sd = as.double(line[[1]][1]); obs_sd = as.double(line[[1]][2])

# BPF data
bpf_mse = strsplit(readLines(raw_bpf_mse, n=1), " ")
bpf_mean_mse_log10 = mean(log10(as.double(bpf_mse[[1]])))
bpf_median_mse_log10 = median(log10(as.double(bpf_mse[[1]])))
bpf_ks = strsplit(readLines(raw_bpf_ks, n=1), " ")
bpf_mean_ks_log10 = mean(log10(as.double(bpf_ks[[1]])))
bpf_median_ks_log10 = median(log10(as.double(bpf_ks[[1]])))

# Multilevel data
line = as.integer(strsplit(readLines(ml_parameters, n=1), " ")[[1]])
N_data = line[1]; N_trials = line[2]; N_ALLOCS = line[3]; N_MESHES = line[4]; N_bpf = line[5]
level0s_str = strsplit(readLines(ml_parameters, n=1), " ")[[1]]
N1s_str = strsplit(readLines(N1s_data, n=1), " ")[[1]]
print(c(N_data, N_trials, N_ALLOCS, N_MESHES, N_bpf))

# Process and clean the MLBPF MSE data for plotting
mse_arr = array(numeric(N_data * N_trials * N_MESHES * N_ALLOCS), dim=c(N_MESHES, N_ALLOCS, N_data * N_trials))
ks_arr = array(numeric(N_data * N_trials * N_MESHES * N_ALLOCS), dim=c(N_MESHES, N_ALLOCS, N_data * N_trials))
for (i_mesh in seq(1, N_MESHES)) {
  for (n_alloc in seq(1, N_ALLOCS)) {
    mse_arr[i_mesh, n_alloc, ] = as.double(strsplit(readLines(raw_mse, n=1), " ")[[1]]) 
    ks_arr[i_mesh, n_alloc, ] = as.double(strsplit(readLines(raw_ks, n=1), " ")[[1]]) 
  }
}
for (n_trial in seq(1, N_data * N_trials)) {
  for (i_mesh in seq(1, N_MESHES)) {
    for (n_alloc in seq(1, N_ALLOCS)) {
      if ((mse_arr[i_mesh, n_alloc, n_trial] == -1) || is.nan(mse_arr[i_mesh, n_alloc, n_trial])) {
        mse_arr[i_mesh, n_alloc, n_trial] = NA
      }
      if ((ks_arr[i_mesh, n_alloc, n_trial] == -1) || is.nan(ks_arr[i_mesh, n_alloc, n_trial])) {
        ks_arr[i_mesh, n_alloc, n_trial] = NA
      }
    }
  }
}

# Construct the dataframe
nx0 <- rep(factor(level0s_str, level0s_str), each=N_data * N_trials)
N1s_rep_tot = rep(factor(x=N1s_str, levels=N1s_str), each=N_MESHES * N_data * N_trials)
data = array(numeric(N_data * N_trials * N_MESHES * N_ALLOCS), dim=c(N_MESHES * N_data * N_trials, N_ALLOCS))
ks_data = array(numeric(N_data * N_trials * N_MESHES * N_ALLOCS), dim=c(N_MESHES * N_data * N_trials, N_ALLOCS))
for (i in seq(1, N_MESHES)) {
  data[as.integer((i - 1) * N_data * N_trials + 1):as.integer(i * N_data * N_trials), ] = aperm(mse_arr[i, , ], perm=c(2, 1))
  ks_data[as.integer((i - 1) * N_data * N_trials + 1):as.integer(i * N_data * N_trials), ] = aperm(ks_arr[i, , ], perm=c(2, 1))
}
log10_mse = c(log10(data))
log10_ks = c(log10(ks_data))
df <- data.frame(N1 = N1s_rep_tot, nx0, log10_mse)
ks_df <- data.frame(N1 = N1s_rep_tot, nx0, log10_ks)
ks_df <- data.frame(N1 = N1s_rep_tot, nx0, log10_ks)
print(tail(ks_df))

single.bp.width <- 1
N.bps <- length(unique(df$nx0))
#N.bps <- length(unique(ks_df$nx0))
bp.hoffsets <- seq(-single.bp.width * (N.bps - (N.bps + 1) / 2 ),by=single.bp.width, length=N.bps)
#df$hoffset <- as.numeric(as.character(df$N1)) + bp.hoffsets[as.numeric(nx0)]
ks_df$hoffset <- as.numeric(as.character(ks_df$N1)) + bp.hoffsets[as.numeric(nx0)]

# Produce the boxplots
#bp <- ggplot(df, mapping=aes(x=hoffset, y=log10_mse, fill=nx0, group=hoffset)) + 
#  geom_tufteboxplot(median.type = "line", 
#                    whisker.type = "line",
#                    hoffset = 0, width=3, size=.5, outlier.size=100) +
#  geom_hline(aes(yintercept=bpf_median_mse_log10, linetype="BPF")) +
#  scale_x_continuous(breaks=seq(0,1000,by=100)) +
#  scale_linetype_manual(name="", values=2) +
#  xlab(bquote(~N[1])) + 
#  ylab(bquote(~log[10](mse))) + 
#  ggtitle("Logarithmic MSE from reference solution") +
#  theme(aspect.ratio=1/2, plot.title=element_text(hjust=0.5),legend.position = "none")
bp <- ggplot(ks_df, mapping=aes(x=hoffset, y=log10_ks, fill=nx0, group=hoffset)) + 
  geom_tufteboxplot(median.type = "line", 
                    whisker.type = "line",
                    hoffset = 0, width=2, size=.5, outlier.size=100) +
  geom_hline(aes(yintercept=bpf_median_ks_log10, linetype="BPF")) + 
  scale_x_continuous(breaks=seq(0,1000,by=100)) + # by=25
  scale_linetype_manual(name="", values=2) +
  xlab(bquote(~N[1])) + 
  ylab(bquote(~log[10](ks))) + 
  ggtitle("Logarithmic KS from reference solution") + 
  theme(aspect.ratio=1/2, plot.title=element_text(hjust=0.5),legend.position = "none")
plot(bp)

# Close the files
close(hmm_data)
close(raw_bpf_mse)
close(raw_bpf_ks)
close(ml_parameters)
close(N1s_data)
close(raw_mse)
close(raw_ks)



