if (!require("pacman")) install.packages("pacman")
pacman::p_load("data.table", "ggplot2", "arrow")


# go to downloads folder
setwd("results/data")

df_raw <- data.table(read_parquet("df_final.parquet"))
df <- df_raw[as.Date(created_at_month_year)>=as.Date("2021/6/23")]
df[, time_to_treat := as.numeric(round(
  difftime(created_at_month_year, as.Date(c("2022/6/28")), units = "weeks")/4))]
df[,commit_count_merged:=commit_count_pr_merged+commit_count_push]
df[,commit_count_merged_main:=commit_count_pr_merged+commit_count_push_main]
df[,commit_count:=commit_count_pr+commit_count_push]
df[,commit_changes_sum:=commit_changes_sum_pr+commit_changes_sum_push]
df[,commit_additions_sum:=commit_additions_sum_pr+commit_additions_sum_push]
df[,commit_deletions_sum:=commit_deletions_sum_pr+commit_deletions_sum_push]
df[,commit_files_changed_sum:=commit_files_changed_sum_pr+commit_files_changed_sum_push]
df[, commit_changes_success:=commit_changes_pr_merged_sum+commit_changes_push_main_sum]
df[, commit_added_success:=commit_additions_pr_merged_sum+commit_additions_push_main_sum]
df[, commit_deleted_success:=commit_deletions_pr_merged_sum+commit_deletions_push_main_sum]
df[, `Treatment: Free Copilot Access`:=treatment]
df[, `Months before Copilot's Release`:=time_to_treat]
df[, actor_repo := paste(actor_id, repo_id, sep = "_")]




df<-df[df$repos_2000 == TRUE]
commit_count_pr

df_pr <- df[commit_changes_sum_pr != 0 & commit_changes_sum_pr  != commit_changes_pr_merged_sum & commit_changes_pr_merged_sum != 0]
df_pr[, pre_freq := sum(as.Date(created_at_month_year)<=as.Date("2022/6/23")), by=c("actor_id")]
df_pr[, post_freq := sum(as.Date(created_at_month_year)>=as.Date("2022/6/23")), by=c("actor_id")]
df_pr <- df_pr[pre_freq >=1 & post_freq>=1]

df_push <- df[commit_changes_sum_push != 0 & commit_changes_sum_push  != commit_changes_push_main_sum & commit_changes_push_main_sum != 0]
df_push[, pre_freq := sum(as.Date(created_at_month_year)<=as.Date("2022/6/23")), by=c("actor_id")]
df_push[, post_freq := sum(as.Date(created_at_month_year)>=as.Date("2022/6/23")), by=c("actor_id")]
df_push <- df_push[pre_freq >=1 & post_freq>=1]

df_commit <- df[commit_changes_sum != 0 & commit_changes_success != 0]
df_commit[, pre_freq := sum(as.Date(created_at_month_year)<=as.Date("2022/6/23")), by=c("actor_id")]
df_commit[, post_freq := sum(as.Date(created_at_month_year)>=as.Date("2022/6/23")), by=c("actor_id")]
df_commit <- df_commit[pre_freq >=1 & post_freq>=1]


df_push_count <- df[push_commit_changes_total!=0 & push_main_commit_changes!=0] 
df_push_count[, pre_freq := sum(as.Date(created_at_month_year)<=as.Date("2022/6/23")), by=c("actor_id")]
df_push_count[, post_freq := sum(as.Date(created_at_month_year)>=as.Date("2022/6/23")), by=c("actor_id")]
df_push_count <- df_push_count[pre_freq >=1 & post_freq>=1]

df_pr_count <- df[!is.na(pr_count) & !is.na(merged_pr_count) & pr_commit_changes != 0 & merged_pr_commit_changes != 0] 
df_pr_count[, pre_freq := sum(as.Date(created_at_month_year)<=as.Date("2022/6/23")), by=c("actor_id")]
df_pr_count[, post_freq := sum(as.Date(created_at_month_year)>=as.Date("2022/6/23")), by=c("actor_id")]
df_pr_count <- df_pr_count[pre_freq >=1 & post_freq>=1]

df_pr_lengths <- df[!is.na(avg_hours_per_pr)]
df_pr_lengths[, pre_freq := sum(as.Date(created_at_month_year)<=as.Date("2022/6/23")), by=c("actor_id")]
df_pr_lengths[, post_freq := sum(as.Date(created_at_month_year)>=as.Date("2022/6/23")), by=c("actor_id")]
df_pr_lengths <- df_pr_lengths[pre_freq >=1 & post_freq>=1]

df_push_lengths <- df[!is.na(avg_hours_per_push)]
df_push_lengths[, pre_freq := sum(as.Date(created_at_month_year)<=as.Date("2022/6/23")), by=c("actor_id")]
df_push_lengths[, post_freq := sum(as.Date(created_at_month_year)>=as.Date("2022/6/23")), by=c("actor_id")]
df_push_lengths <- df_push_lengths[pre_freq >=1 & post_freq>=1]


commit_count_pr_mod <- feols(
  commit_count_pr ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id, df_pr,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pr_commits.png", width = 720, height = 480, res = 120)
coefplot(commit_count_pr_mod, keep = ":", 
         value.lab = "Added PR Commits",
         xlab = "Months before Copilot's Release\n",
         main = "Impact of Copilot's Release on PR Commits",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

commit_count_pr_mod_weight <- feols(
  commit_count_pr_weight ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_pr,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pr_commits_weight.png", width = 720, height = 480, res = 120)
coefplot(commit_count_pr_mod_weight, keep = ":", 
         value.lab = "Added PR Commits (Weighted)",
         xlab = "Months before Copilot's Release\n",
         main = "Impact of Copilot's Release on PR Commits",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer\nWeight: Changes Made in Commit"
)
abline(v = 12, col = "blue")
dev.off()

commit_count_pr_merged_mod <- feols(
  commit_count_pr_merged ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_pr,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pr_merged_commits.png", width = 720, height = 480, res = 120)
coefplot(commit_count_pr_merged_mod, keep = ":", 
         value.lab = "Added Merged PR Commits",
         xlab = "Months before Copilot's Release\n",
         main = "Impact of Copilot's Release on Merged PR Commits",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

commit_count_pr_merged_mod_weight <- feols(
  commit_count_pr_merged ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_pr,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pr_merged_commits_weight.png", width = 720, height = 480, res = 120)
coefplot(commit_count_pr_merged_mod_weight, keep = ":", 
         value.lab = "Added Merged PR Commits (Weighted)",
         xlab = "Months before Copilot's Release\n",
         main = "Impact of Copilot's Release on Merged PR Commits",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer\nWeight: Changes Made in Commit"
)
abline(v = 12, col = "blue")
dev.off()

commit_count_pr_notmerged_mod <- feols(
  unmerged_pr_commits ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_pr,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pr_notmerged_commits.png", width = 720, height = 480, res = 120)
coefplot(commit_count_pr_notmerged_mod, keep = ":", 
         value.lab = "Added Not Merged PR Commits",
         xlab = "Months before Copilot's Release\n",
         main = "Impact of Copilot's Release on Not Merged PR Commits",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

commit_count_pr_notmerged_mod_weight <- feols(
  unmerged_pr_commits_weight  ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_pr,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pr_notmerged_commits_weight.png", width = 720, height = 480, res = 120)
coefplot(commit_count_pr_notmerged_mod_weight, keep = ":", 
         value.lab = "Added Not Merged PR Commits (Weighted)",
         xlab = "Months before Copilot's Release\n",
         main = "Impact of Copilot's Release on Not Merged PR Commits",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer\nWeight: Changes Made in Commit"
)
abline(v = 12, col = "blue")
dev.off()


commit_count_push_mod <- feols(
  commit_count_push ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_push, cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_push_commits.png", width = 720, height = 480, res = 120)
coefplot(commit_count_push_mod, keep = ":", 
         value.lab = "Added Push Commits",
         xlab = "Months before Copilot's Release\n",
         main = "Impact of Copilot's Release on Push Commits",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

commit_count_push_mod_weight <- feols(
  commit_count_push_weight ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_push, cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_push_commits_weight.png", width = 720, height = 480, res = 120)
coefplot(commit_count_push_mod_weight, keep = ":", 
         value.lab = "Added Push Commits (Weighted)",
         xlab = "Months before Copilot's Release\n",
         main = "Impact of Copilot's Release on Push Commits",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer\nWeight: Changes Made in Commit"
)
abline(v = 12, col = "blue")
dev.off()

commit_count_push_main_mod <- feols(
  commit_count_push_main ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_push,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_push_main_commits.png", width = 720, height = 480, res = 120)
coefplot(commit_count_push_main_mod, keep = ":", 
         value.lab = "Added Push Commits to Main",
         xlab = "Months before Copilot's Release\n",
         main = "Impact of Copilot's Release on Push Commits to Main",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

commit_count_push_main_mod_weight <- feols(
  commit_count_push_main_weight ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_push,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_push_main_commits_weight.png", width = 720, height = 480, res = 120)
coefplot(commit_count_push_main_mod_weight, keep = ":", 
         value.lab = "Added Push Commits to Main (Weighted)",
         xlab = "Months before Copilot's Release\n",
         main = "Impact of Copilot's Release on Push Commits to Main",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer\nWeight: Changes Made in Commit"
)
abline(v = 12, col = "blue")
dev.off()

commit_count_push_notmain_mod <- feols(
  commit_count_push_notmain ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release` + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_push,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_push_notmain_commits.png", width = 720, height = 480, res = 120)
coefplot(commit_count_push_notmain_mod, keep = ":", 
         value.lab = "Added Push Commits to Non-Main",
         xlab = "Months before Copilot's Release\n",
         main = "Impact of Copilot's Release on Push Commits to Non-Main",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

commit_count_push_notmain_mod_weight <- feols(
  commit_count_push_notmain_weight ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_push,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_push_notmain_commits_weight.png", width = 720, height = 480, res = 120)
coefplot(commit_count_push_notmain_mod_weight, keep = ":", 
         value.lab = "Added Push Commits to Non-Main (Weighted)",
         xlab = "Months before Copilot's Release\n",
         main = "Impact of Copilot's Release on Push Commits to Non-Main",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer\nWeight: Changes Made in Commit"
)
abline(v = 12, col = "blue")
dev.off()

commit_count_mod <- feols(
  commit_count ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_commit,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_commits.png", width = 720, height = 480, res = 120)
coefplot(commit_count_mod, keep = c(":"), 
         value.lab = "Added Commits",
         xlab = "Months before Copilot's Public Free Release",
         main = "Impact of Copilot's Release on Commit Count",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

commit_count_merged_mod <- feols(
  commit_count_merged_main ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_commit,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_commits_merged.png", width = 720, height = 480, res = 120)
coefplot(commit_count_merged_mod, keep = c(":"), 
         value.lab = "Added Successful Commits",
         xlab = "Months before Copilot's Public Free Release\n",
         main = "Impact of Copilot's Release on Successful Commits Count",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer\nSuccessful: Merged PR Commits or Push Commits to Main/Master"
)
abline(v = 12, col = "blue")
dev.off()


commit_count_mod_weight <- feols(
  commit_count_weight ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_commit,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_commits_count.png", width = 720, height = 480, res = 120)
coefplot(commit_count_mod_weight, keep = c(":"), 
         value.lab = "Added Commits (Weighted)",
         xlab = "Months before Copilot's Release\n",
         main = "Impact of Copilot's Release on Commits",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer\nWeight: Changes Made in Commit"
)
abline(v = 12, col = "blue")
dev.off()

commit_count_mod_weight_add <- feols(
  commit_count_weight_add ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_commit,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_commits_count_add.png", width = 720, height = 480, res = 120)
coefplot(commit_count_mod_weight_add, keep = c(":"), 
         value.lab = "Added Commits (Weighted)",
         xlab = "Months before Copilot's Release\n",
         main = "Impact of Copilot's Release on Commits",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer\nWeight: Additions Made in Commit"
)
abline(v = 12, col = "blue")
dev.off()

commit_count_mod_weight_delete <- feols(
  commit_count_weight_delete ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_commit,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_commits_count_delete.png", width = 720, height = 480, res = 120)
coefplot(commit_count_mod_weight_delete, keep = c(":"), 
         value.lab = "Added Commits (Weighted)",
         xlab = "Months before Copilot's Release\n",
         main = "Impact of Copilot's Release on Commits",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer\nWeight: Deletions Made in Commit"
)
abline(v = 12, col = "blue")
dev.off()

commit_count_mod_weight <- feols(
  commit_count_merged_main_weight ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_commit,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_commits_merged_count.png", width = 720, height = 480, res = 120)
coefplot(commit_count_mod_weight, keep = c(":"), 
         value.lab = "Added Succesful Commits (Weighted)",
         xlab = "Months before Copilot's Release\n\n",
         main = "Impact of Copilot's Release on Successful Commits",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer\nWeight: Changes Made in Commit\nSuccessful: Merged PR Commits or Push Commits to Main/Master"
)
abline(v = 12, col = "blue")
dev.off()

commit_count_mod_weight_add <- feols(
  commit_count_merged_main_weight_add ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release` + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_commit,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_commits_merged_count_add.png", width = 720, height = 480, res = 120)
coefplot(commit_count_mod_weight_add, keep = c(":"), 
         value.lab = "Added Succesful Commits (Weighted)",
         xlab = "Months before Copilot's Release\n\n",
         main = "Impact of Copilot's Release on Successful Commits",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer\nWeight: Additions Made in Commit\nSuccessful: Merged PR Commits or Push Commits to Main/Master"
)
abline(v = 12, col = "blue")
dev.off()

commit_count_mod_weight_delete <- feols(
  commit_count_merged_main_weight_delete ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_commit,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_commits_merged_count_delete.png", width = 720, height = 480, res = 120)
coefplot(commit_count_mod_weight_delete, keep = c(":"), 
         value.lab = "Added Succesful Commits (Weighted)",
         xlab = "Months before Copilot's Release\n\n",
         main = "Impact of Copilot's Release on Successful Commits",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer\nWeight: Deletions Made in Commit\nSuccessful: Merged PR Commits or Push Commits to Main/Master"
)
abline(v = 12, col = "blue")
dev.off()


push_count <- feols(
  push_count ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_push_count,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pushes.png", width = 720, height = 480, res = 120)
coefplot(push_count, keep = c(":"), 
         value.lab = "Added Pushes",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on Push Count",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

push_count_weight <- feols(
  push_count_weight ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_push_count,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pushes_weight.png", width = 720, height = 480, res = 120)
coefplot(push_count_weight, keep = c(":"), 
         value.lab = "Added Pushes",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on Push Count",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer\nWeight: Changes Made in Push"
)
abline(v = 12, col = "blue")
dev.off()

push_non_mainpush_main_count_count <- feols(
  push_main_count ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_push_count,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pushes_main.png", width = 720, height = 480, res = 120)
coefplot(push_main_count, keep = c(":"), 
         value.lab = "Added Pushes (to Main Branch)",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on Main Branch Push Count",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

push_non_main_count <- feols(
  push_main_count_weight ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_push_count,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pushes_main_weight.png", width = 720, height = 480, res = 120)
coefplot(push_main_count, keep = c(":"), 
         value.lab = "Added Pushes (to Main Branch)",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on Main Branch Push Count\nWeight: Changes Made in Commit",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

push_non_main_count <- feols(
  push_notmain_count ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_push_count,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pushes_nonmain.png", width = 720, height = 480, res = 120)
coefplot(push_non_main_count, keep = c(":"), 
         value.lab = "Added Pushes (to Non-Main Branches)",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on Non-Main Branches Push Count",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

push_non_main_count <- feols(
  push_notmain_count ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_push_count,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pushes_nonmain_weight.png", width = 720, height = 480, res = 120)
coefplot(push_non_main_count, keep = c(":"), 
         value.lab = "Added Pushes (to Non-Main Branches)",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on Non-Main Branches Push Count\nWeight: Changes Made in Commit",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()


pr_count <- feols(
  pr_count ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_pr_count,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pr.png", width = 720, height = 480, res = 120)
coefplot(pr_count, keep = c(":"), 
         value.lab = "Added Pull Requests Opened",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on PRs Opened",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

pr_count_weight <- feols(
  pr_count_weight ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_pr_count,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pr_weight.png", width = 720, height = 480, res = 120)
coefplot(pr_count_weight, keep = c(":"), 
         value.lab = "Added Pull Requests Opened",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on PRs Opened",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

pr_count_merged <- feols(
  merged_pr_count ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_pr_count,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pr_merged.png", width = 720, height = 480, res = 120)
coefplot(pr_count_merged, keep = c(":"), 
         value.lab = "Added Pull Requests Merged",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on PRs Merged",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

pr_count_merged_weight <- feols(
  merged_pr_count_weight ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_pr_count,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pr_merged_weights.png", width = 720, height = 480, res = 120)
coefplot(pr_count_merged_weight, keep = c(":"), 
         value.lab = "Added Pull Requests Merged",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on PRs Merged",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()


pr_count_nonmerged <- feols(
    pr_notmerged_count  ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_pr_count,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pr_nonmerged.png", width = 720, height = 480, res = 120)
coefplot(pr_count_nonmerged, keep = c(":"), 
         value.lab = "Added Unmerged Pull Requests ",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on Unmerged PRs",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

pr_count_nonmerged_weight <- feols(
  pr_notmerged_count_weight ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_pr_count,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pr_nonmerged_weights.png", width = 720, height = 480, res = 120)
coefplot(pr_count_nonmerged_weight, keep = c(":"), 
         value.lab = "Added Unmerged Pull Requests",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on Unmerged PRs",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()


pr_length <- feols(
  avg_hours_per_pr ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_pr_lengths,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pr_length.png", width = 720, height = 480, res = 120)
coefplot(pr_length, keep = c(":"), 
         value.lab = "Added Hours to PR Merging",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on PR Merge Time",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

pr_length <- feols(
  avg_hours_per_pr ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_pr_lengths,
  cluster = c("actor_id"), fixef.rm = "both",
  weights = df_pr_lengths$pr_length_commit_changes)
png("../copilot/copilot_pr_length_weight.png", width = 720, height = 480, res = 120)
coefplot(pr_length, keep = c(":"), 
         value.lab = "Added Hours to PR Merging",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on PR Merge Time",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()


pr_length <- feols(
  avg_hours_per_pr_commit ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_pr_lengths,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pr_length_commit_norm.png", width = 720, height = 480, res = 120)
coefplot(pr_length, keep = c(":"), 
         value.lab = "Added Hours/Commit to PR Merging",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on PR Merge Time",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

pr_length <- feols(
  avg_hours_per_pr_commit ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_pr_lengths,
  cluster = c("actor_id"), fixef.rm = "both",
  weights = df_pr_lengths$pr_length_commit_changes)
png("../copilot/copilot_pr_length_weight_commit_norm.png", width = 720, height = 480, res = 120)
coefplot(pr_length, keep = c(":"), 
         value.lab = "Added Hours/Commit to PR Merging",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on PR Merge Time",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()



pr_length <- feols(
  avg_hours_per_pr_commit_change ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_pr_lengths,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pr_length_change_norm.png", width = 720, height = 480, res = 120)
coefplot(pr_length, keep = c(":"), 
         value.lab = "Added Hours/Change to PR Merging",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on PR Merge Time",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

pr_length <- feols(
  avg_hours_per_pr_commit_change ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_pr_lengths,
  cluster = c("actor_id"), fixef.rm = "both",
  weights = df_pr_lengths$pr_length_commit_changes)
png("../copilot/copilot_pr_length_weight_change_norm.png", width = 720, height = 480, res = 120)
coefplot(pr_length, keep = c(":"), 
         value.lab = "Added Hours/Change to PR Merging",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on PR Merge Time",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()


pr_length <- feols(
  pr_code_hours ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_pr_lengths,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pr_length_code.png", width = 720, height = 480, res = 120)
coefplot(pr_length, keep = c(":"), 
         value.lab = "Added Hours to PR Code Completion",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on PR Coding Time",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

pr_length <- feols(
  code_hours_per_pr_commit ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_pr_lengths,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pr_length_code_commit_norm.png", width = 720, height = 480, res = 120)
coefplot(pr_length, keep = c(":"), 
         value.lab = "Added Hours/Commit to PR Code Completion",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on PR Coding Time",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

pr_length <- feols(
  code_hours_per_pr_commit_change ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_pr_lengths,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_pr_length_code_change_norm.png", width = 720, height = 480, res = 120)
coefplot(pr_length, keep = c(":"), 
         value.lab = "Added Hours/Change to PR Code Completion",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on PR Coding Time",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()


push_length <- feols(
  avg_hours_per_push ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_push_lengths,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_push_length.png", width = 720, height = 480, res = 120)
coefplot(push_length, keep = c(":"), 
         value.lab = "Added Hours to Push",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on Push Time",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

push_length <- feols(
  avg_hours_per_push ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_push_lengths,
    weights = df_push_lengths$push_length
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_push_length.png", width = 720, height = 480, res = 120)
coefplot(push_length, keep = c(":"), 
         value.lab = "Added Hours to Push",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on Push Time",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

push_length <- feols(
  avg_hours_per_push_commit ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_push_lengths,
  cluster = c("actor_id"), fixef.rm = "both")
png("../copilot/copilot_push_length_norm_commit.png", width = 720, height = 480, res = 120)
coefplot(push_length, keep = c(":"), 
         value.lab = "Added Hours/Commit to Push",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on Push Time",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

push_length <- feols(
  avg_hours_per_push_commit_change ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`   + cumulative_forks + cumulative_watches + appearances + coworkers + repo_issues_opened + repo_issues_closed + issues_managed + comments_made
  | actor_id^repo_id , df_push_lengths,
  cluster = c("actor_id"))
png("../copilot/copilot_push_length_norm_change.png", width = 720, height = 480, res = 120)
coefplot(push_length, keep = c(":"), 
         value.lab = "Added Hours/Change to Push",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on Push Time",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()
