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
readr::cols_only(subj="c",
                 # pretest="c",
                 intervention="c",
                 timepoint="c",
                 condition="c",
                 time="d",
                 value="d",
                 method="c",
                 roi="c") -> colspec
readr::read_csv("roi-MPM_IOS_IOG_pOTS_lh-timeseries-long.csv",
                col_types=colspec) -> rawdata

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
formula(value ~ cond_ * tmpt_ * intv_ + (1 + cond_ + tmpt_ + intv_ | subj)
        ) -> form
# initial fit w/ Satterthwaite p-values
afex::mixed(form, data=modeldata, method="S", check_contrasts=FALSE) -> mod
# try all optimizers to see if convergence warnings are false alarms
lme4::lmer(form, data=modeldata, control=lme4::lmerControl(optimizer=NULL)) -> emptymod
lme4::allFit(emptymod) -> allfit_results
print(summary(allfit_results))  # all optimizers agree on fixed effect coefs.

# show the ANOVA summary table
print(mod$anova_table)

# show the individual coefficients
print(summary(mod))

# this will give post-hoc faces-minus-words & cars-minus-words for each timepoint and intervention group
emmeans::emmeans(mod, "cond_", by=c("tmpt_", "intv_"), type="response",
                 contr="trt.vs.ctrl") ->
    post_hoc_conds
print(post_hoc_conds$contrasts)

# this will give post-hoc post-minus-pre for each condition in each intervention group
emmeans::emmeans(mod, "tmpt_", by=c("cond_", "intv_"), type="response",
                 contr="revpairwise") ->
    post_hoc_timepoints
print(post_hoc_timepoints$contrasts)

# Just examine word response and compare changes between intervention groups
formula(value ~ tmpt_ * intv_ + (1 | subj)) -> form
afex::mixed(form, data=filter(modeldata,cond_ == 'words'), method="S", check_contrasts=FALSE) -> mod
print(mod$anova_table)
print(summary(mod))

# Fit model to pre data
formula(value ~ cond_ * intv_ + (1 | subj)) -> form
afex::mixed(form, data=filter(modeldata,tmpt_ == 'pre'), method="S", check_contrasts=FALSE) -> mod
print(mod$anova_table)
print(summary(mod))

# Fit model to post data
formula(value ~ cond_ * intv_ + (1 | subj)) -> form
afex::mixed(form, data=filter(modeldata,tmpt_ == 'post'), method="S", check_contrasts=FALSE) -> mod
print(mod$anova_table)
print(summary(mod))

