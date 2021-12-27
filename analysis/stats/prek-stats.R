library(dplyr)

## function for factor coding
deviation_coding <- function(x, levs=NULL) {
    if(is.null(levs)) levs <- unique(x)
    x <- factor(x, levels=levs)
    dnames <- list(levels(x), levels(x)[2])
    contrasts(x) <- matrix(c(-0.5, 0.5), nrow=2, dimnames=dnames)
    x
}

# read temporal ROI value
yaml::read_yaml(file.path("..", "final-figs", "peak-of-grand-mean.yaml")) %>%
    `[[`("temporal_roi") -> temporal_roi

# load data
yaml::read_yaml(file.path("..", "..", "params", "paths.yaml")) %>%
    `[[`("results_dir") -> results_dir
file.path(results_dir, "original-long-tsss-no-blink-proj", "roi",
          "time-series", "roi-MPM_IOS_IOG_pOTS_lh-timeseries-long.csv") ->
          csv_path
# uncomment if you have a local copy of the data:
# "roi-MPM_IOS_IOG_pOTS_lh-timeseries-long.csv" -> csv_path

readr::cols_only(subj="c",
                 # pretest="c",
                 intervention="c",
                 timepoint="c",
                 condition="c",
                 time="d",
                 value="d",
                 method="c",
                 roi="c") -> colspec
readr::read_csv(csv_path, col_types=colspec) -> rawdata

# prepare for modeling
rawdata %>%
    filter(method %in% "dSPM",
           roi %in% "MPM_IOS_IOG_pOTS_lh",
           condition %in% c("words", "faces", "cars"),
           temporal_roi[1] <= time,
           temporal_roi[2] >= time) %>%
    mutate(cond_=factor(condition, levels=c("words", "faces", "cars")),
           tmpt_=deviation_coding(.$timepoint, levs=c("pre", "post")),
           intv_=deviation_coding(.$intervention, levs=c("language", "letter"))) %>%
    group_by(cond_, tmpt_, intv_, subj) %>%
    summarise(value=mean(value), .groups="keep") ->
    modeldata

matrix(c(0, 1, 0, 0, 0, 1), nrow=3, ncol=2, byrow=FALSE,
       dimnames=list(c("words", "faces", "cars"), c("faces", "cars"))) ->
    contrasts(modeldata$cond_)

# model
formula(value ~ cond_ * tmpt_ * intv_ + (1 + cond_ + tmpt_ | subj)
        ) -> form_full
sink("results-log.txt")
# initial fit w/ parametric bootstrapped p-values
afex::mixed(form_full, data=modeldata, method="PB", check_contrasts=FALSE
            ) -> fullmod
# try all optimizers to see if convergence warnings are false alarms
cat("\n# # # # ALLFIT RESULTS # # # #\n")
lme4::lmer(form_full, data=modeldata, control=lme4::lmerControl(optimizer=NULL)
           ) -> emptymod
lme4::allFit(emptymod) -> allfit_results
print(summary(allfit_results))  # all optimizers agree on fixed effect coefs.

# show the ANOVA summary table
cat("\n# # # # ANOVA TABLE # # # #\n")
print(fullmod$anova_table)

# show the individual coefficients
cat("\n# # # # MODEL SUMMARY # # # #\n")
print(summary(fullmod))

# this will give post-hoc faces-minus-words & cars-minus-words for each
# timepoint and intervention group
cat("\n# # # # POST-HOC COMPARISONS: CONDITIONS # # # #\n")
emmeans::emmeans(fullmod, "cond_", by=c("tmpt_", "intv_"), type="response",
                 contr="trt.vs.ctrl") ->
    post_hoc_conds
print(post_hoc_conds$contrasts)

# this will give post-hoc post-minus-pre for each condition in each
# intervention group
cat("\n# # # # POST-HOC COMPARISONS: POST-MINUS-PRE # # # #\n")
emmeans::emmeans(fullmod, "tmpt_", by=c("cond_", "intv_"), type="response",
                 contr="revpairwise") ->
    post_hoc_timepoints
print(post_hoc_timepoints$contrasts)

# Fit model to pre data
formula(value ~ cond_ * intv_ + (1 | subj)
        ) -> form
cat("\n# # # # PRE-CAMP ONLY MODEL # # # #\n")
afex::mixed(form, data=filter(modeldata, tmpt_=="pre"), method="PB",
            check_contrasts=FALSE) -> premod
print(premod$anova_table)
print(summary(premod))

# Fit model to post data
cat("\n# # # # POST-CAMP ONLY MODEL # # # #\n")
afex::mixed(form, data=filter(modeldata, tmpt_=="post"), method="PB",
            check_contrasts=FALSE) -> postmod
print(postmod$anova_table)
print(summary(postmod))
sink()
