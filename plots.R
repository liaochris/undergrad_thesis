if (!require("pacman")) install.packages("pacman")
pacman::p_load("data.table", "ggplot2", "did", "fixest")


# go to downloads folder
setwd("~/undergrad_thesis/results/data")

df_raw <- fread("df_copilot_commits_grouped.csv")
df <- df_raw[as.Date(created_at_month_year)>=as.Date("2021/6/23") & as.Date(created_at_month_year)<=as.Date("2023/6/23")]
df$V1 <- NULL
df$group <- 1
df[, time_to_treat := as.factor(round(
  difftime(created_at_month_year, as.Date(c("2022/6/28")), units = "weeks")/4))]
df[,commit_count_merged:=pr_merged_sum+commit_count_push]
df[,commit_count_merged_main:=pr_merged_sum+push_main_sum]
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
df <- df[df$greater_500_stars %in% c(TRUE)]
# df <- df[df$greater_500_stars %in% c(TRUE)]


# + `gained stars` + `gained forks` +
#`cumulative stars` + `cumulative forks` + `opened issues` + `opened prs` +
#`cumulative opened issues` + `cumulative opened prs` + `added triagers` +
#`added writers` + `cumulative triagers`  + `cumulative writers`  +`commit count` + `contributed_projects` + `user opened issues`+ `user #opened prs` +
#`cumulative user opened issues` + `cumulative user opened prs` + `user added triagers` +
#`user added writers` + `user cumulative triagers` + `user cumulative writers`

commit_count_pr_mod <- feols(
  commit_count_pr ~ `Treatment: Free Copilot Access` + `Months before Copilot's Release` + `Treatment: Free Copilot Access`:`Months before Copilot's Release`
 | commit_actor_id^repo_id , df,
  cluster = c("commit_actor_id"))
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
  commit_count_pr ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

    | commit_actor_id^repo_id , df,
  cluster = c("commit_actor_id"),
  weights = df$commit_changes_sum_pr)
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
  pr_merged_sum ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | commit_actor_id^repo_id , df,
  cluster = c("commit_actor_id"))
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
  pr_merged_sum ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | commit_actor_id^repo_id , df,
  cluster = c("commit_actor_id"),
 weights = df$commit_changes_pr_merged_sum)
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
  commit_count_pr - pr_merged_sum ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | commit_actor_id^repo_id , df,
  cluster = c("commit_actor_id"))
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
  commit_count_pr - pr_merged_sum  ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | commit_actor_id^repo_id , df,
  cluster = c("commit_actor_id"),
    weight = df$commit_changes_sum_pr - df$commit_changes_pr_merged_sum)
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
  commit_count_push ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`
 | 
    commit_actor_id^repo_id , df, cluster = c("commit_actor_id"))
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
  commit_count_push ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`
 | 
    commit_actor_id^repo_id , df, cluster = c("commit_actor_id"), 
    weights = df$commit_changes_sum_push)
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
  push_main_sum ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`
 | 
    commit_actor_id^repo_id , df,
  cluster = c("commit_actor_id"))
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
  push_main_sum ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`
 | 
    commit_actor_id^repo_id , df,
  cluster = c("commit_actor_id"),
    weights = df$commit_changes_push_main_sum)
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
  commit_count_push - push_main_sum ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`
 | 
    commit_actor_id^repo_id , df,
  cluster = c("commit_actor_id"))
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
  commit_count_push - push_main_sum ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`
 | 
    commit_actor_id^repo_id , df,
  cluster = c("commit_actor_id"),
    weights = df$commit_changes_sum_push - df$commit_changes_push_main_sum)
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
  commit_count ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | commit_actor_id^repo_id , df,
  cluster = c("commit_actor_id"))
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
  commit_count_merged_main ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | commit_actor_id^repo_id , df,
  cluster = c("commit_actor_id"))
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
  commit_count ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | commit_actor_id^repo_id , df,
  cluster = c("commit_actor_id"),
  weights = df$commit_changes_sum)
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
  commit_count ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | commit_actor_id^repo_id , df,
  cluster = c("commit_actor_id"),
  weights = df$commit_additions_sum)
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
  commit_count ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | commit_actor_id^repo_id , df,
  cluster = c("commit_actor_id"),
  weights = df$commit_deletions_sum)
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
  commit_count_merged_main ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | commit_actor_id^repo_id , df,
  cluster = c("commit_actor_id"),
  weights = df$commit_changes_success)
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
  commit_count_merged_main ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | commit_actor_id^repo_id , df,
  cluster = c("commit_actor_id"),
  weights = df$commit_added_success)
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
  commit_count_merged_main ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | commit_actor_id^repo_id , df,
  cluster = c("commit_actor_id"),
  weights = df$commit_deleted_success)
png("../copilot/copilot_commits_merged_count_delete.png", width = 720, height = 480, res = 120)
coefplot(commit_count_mod_weight_delete, keep = c(":"), 
         value.lab = "Added Succesful Commits (Weighted)",
         xlab = "Months before Copilot's Release\n\n",
         main = "Impact of Copilot's Release on Successful Commits",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer\nWeight: Deletions Made in Commit\nSuccessful: Merged PR Commits or Push Commits to Main/Master"
)
abline(v = 12, col = "blue")
dev.off()


df_push <- fread("df_copilot_push_grouped.csv")
df_push <- df_push[as.Date(created_at_month_year)>=as.Date("2021/6/23") & as.Date(created_at_month_year)<=as.Date("2023/6/23")]
df_push[, time_to_treat := as.factor(round(
  difftime(created_at_month_year, as.Date(c("2022/6/28")), units = "weeks")/4))]
df_push[, `Treatment: Free Copilot Access`:=treatment]
df_push[, `Months before Copilot's Release`:=time_to_treat]
df_push <- df_push[df_push$greater_500_stars %in% c(TRUE)]
# df_push <- df_push[df_push$greater_500_stars %in% c(TRUE)]


push_count <- feols(
  push_count ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | commit_actor_id^repo_id , df_push,
  cluster = c("commit_actor_id"))
png("../copilot/copilot_pushes.png", width = 720, height = 480, res = 120)
coefplot(push_count, keep = c(":"), 
         value.lab = "Added Pushes",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on Push Count",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

push_main_count <- feols(
  push_main_count ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | commit_actor_id^repo_id , df_push,
  cluster = c("commit_actor_id"))
png("../copilot/copilot_pushes_main.png", width = 720, height = 480, res = 120)
coefplot(push_main_count, keep = c(":"), 
         value.lab = "Added Pushes (to Main Branch)",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on Main Branch Push Count",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()


df_pr <- fread("df_copilot_prs_grouped.csv")
df_pr <- df_pr[as.Date(created_at_month_year)>=as.Date("2021/6/23") & as.Date(created_at_month_year)<=as.Date("2023/6/23")]
df_pr[, time_to_treat := as.factor(round(
  difftime(created_at_month_year, as.Date(c("2022/6/28")), units = "weeks")/4))]
df_pr[, `Treatment: Free Copilot Access`:=treatment]
df_pr[, `Months before Copilot's Release`:=time_to_treat]
df_pr <- df_pr[df_pr$greater_500_stars %in% c(TRUE)]
# df_pr <- df_pr[df_pr$greater_500_stars %in% c(TRUE)]


pr_count <- feols(
  pr_number_count ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | pr_user_id^repo_id , df_pr,
  cluster = c("pr_user_id"))
png("../copilot/copilot_pr.png", width = 720, height = 480, res = 120)
coefplot(pr_count, keep = c(":"), 
         value.lab = "Added Pull Requests Opened",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on PRs Opened",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

pr_count_merged <- feols(
  merged_pr_sum ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | pr_user_id^repo_id , df_pr,
  cluster = c("pr_user_id"))
png("../copilot/copilot_pr_merged.png", width = 720, height = 480, res = 120)
coefplot(pr_count_merged, keep = c(":"), 
         value.lab = "Added Pull Requests Merged",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on PRs Merged",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

df_pr_lengths <- fread("df_copilot_pr_lengths_grouped.csv")
df_pr_lengths <- df_pr_lengths[as.Date(created_at_month_year)>=as.Date("2021/6/23") & as.Date(created_at_month_year)<=as.Date("2023/6/23")]
df_pr_lengths[, time_to_treat := as.factor(round(
  difftime(created_at_month_year, as.Date(c("2022/6/28")), units = "weeks")/4))]
df_pr_lengths[, `Treatment: Free Copilot Access`:=treatment]
df_pr_lengths[, `Months before Copilot's Release`:=time_to_treat]
df_pr_lengths <- df_pr_lengths[df_pr_lengths$greater_500_stars %in% c(TRUE)]
# df_pr_lengths <- df_pr_lengths[df_pr_lengths$greater_500_stars %in% c(TRUE)]



pr_length <- feols(
  hours ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | actor_id^repo_id , df_pr_lengths,
  cluster = c("actor_id"))
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
  hours_per_commit ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | actor_id^repo_id , df_pr_lengths,
  cluster = c("actor_id"))
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
  hours_per_commit_change ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | actor_id^repo_id , df_pr_lengths,
  cluster = c("actor_id"))
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
  code_hours ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | actor_id^repo_id , df_pr_lengths,
  cluster = c("actor_id"))
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
  code_hours_per_commit ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | actor_id^repo_id , df_pr_lengths,
  cluster = c("actor_id"))
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
  code_hours_per_commit_change ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | actor_id^repo_id , df_pr_lengths,
  cluster = c("actor_id"))
png("../copilot/copilot_pr_length_code_change_norm.png", width = 720, height = 480, res = 120)
coefplot(pr_length, keep = c(":"), 
         value.lab = "Added Hours/Change to PR Code Completion",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on PR Coding Time",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

df_push_lengths <- fread("df_copilot_push_lengths_grouped.csv")
df_push_lengths <- df_push_lengths[as.Date(created_at_month_year)>=as.Date("2021/6/23") & as.Date(created_at_month_year)<=as.Date("2023/6/23")]
df_push_lengths[, time_to_treat := as.factor(round(
  difftime(created_at_month_year, as.Date(c("2022/6/28")), units = "weeks")/4))]
df_push_lengths[, `Treatment: Free Copilot Access`:=treatment]
df_push_lengths[, `Months before Copilot's Release`:=time_to_treat]
df_push_lengths <- df_push_lengths[df_push_lengths$greater_500_stars %in% c(TRUE)]
# df_push_lengths <- df_push_lengths[df_push_lengths$greater_500_stars %in% c(TRUE)]



push_length <- feols(
  hours ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | actor_id^repo_id , df_push_lengths,
  cluster = c("actor_id"))
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
  hours_per_commit ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | actor_id^repo_id , df_push_lengths,
  cluster = c("actor_id"))
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
  hours_per_commit_change ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

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

df_push_lengths_main <- fread("df_copilot_push_lengths_main_grouped.csv")
df_push_lengths_main <- df_push_lengths_main[as.Date(created_at_month_year)>=as.Date("2021/6/23") & as.Date(created_at_month_year)<=as.Date("2023/6/23")]
df_push_lengths_main[, time_to_treat := as.factor(round(
  difftime(created_at_month_year, as.Date(c("2022/6/28")), units = "weeks")/4))]
df_push_lengths_main[, `Treatment: Free Copilot Access`:=treatment]
df_push_lengths_main[, `Months before Copilot's Release`:=time_to_treat]
df_push_lengths_main <- df_push_lengths_main[df_push_lengths_main$greater_500_stars %in% c(TRUE)]
# df_push_lengths_main <- df_push_lengths_main[df_push_lengths_main$greater_500_stars %in% c(TRUE)]


push_length_main <- feols(
  hours ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | actor_id^repo_id , df_push_lengths_main,
  cluster = c("actor_id"))
png("../copilot/copilot_push_main_length.png", width = 720, height = 480, res = 120)
coefplot(push_length_main, keep = c(":"), 
         value.lab = "Added Hours to Push to Main",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on Push (Main Branch) Time",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()

df_push_lengths_notmain <- fread("df_copilot_push_lengths_notmain_grouped.csv")
df_push_lengths_notmain <- df_push_lengths_notmain[as.Date(created_at_month_year)>=as.Date("2021/6/23") & as.Date(created_at_month_year)<=as.Date("2023/6/23")]
df_push_lengths_notmain[, time_to_treat := as.factor(round(
  difftime(created_at_month_year, as.Date(c("2022/6/28")), units = "weeks")/4))]
df_push_lengths_notmain[, `Treatment: Free Copilot Access`:=treatment]
df_push_lengths_notmain[, `Months before Copilot's Release`:=time_to_treat]
df_push_lengths_notmain <- df_push_lengths_notmain[df_push_lengths_notmain$greater_500_stars %in% c(TRUE)]
# df_push_lengths_notmain <- df_push_lengths_notmain[df_push_lengths_notmain$greater_500_stars %in% c(TRUE)]


push_length_notmain <- feols(
  hours ~ `Treatment: Free Copilot Access`*`Months before Copilot's Release`

  | actor_id^repo_id , df_push_lengths_notmain,
  cluster = c("actor_id"))
png("../copilot/copilot_push_notmain_length.png", width = 720, height = 480, res = 120)
coefplot(push_length_notmain, keep = c(":"), 
         value.lab = "Added Hours to Push to Non-Main",
         xlab = "Months before Copilot's Release",
         main = "Impact of Copilot's Release on Push (Non-Main Branch) Time",
         sub = "Treatment: Free Copilot as Top Open Source Maintainer"
)
abline(v = 12, col = "blue")
dev.off()
