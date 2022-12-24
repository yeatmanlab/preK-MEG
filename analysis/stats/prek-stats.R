library(dplyr)
library(ggplot2)

## function for factor coding
deviation_coding <- function(x, levs=NULL) {
    if (is.null(levs)) levs <- unique(x)
    x <- factor(x, levels=levs)
    dnames <- list(levels(x), levels(x)[2])
    contrasts(x) <- matrix(c(-0.5, 0.5), nrow=2, dimnames=dnames)
    x
}

# select our ROI. options are:
#     CoS_lh, CoS_rh, mFus_lh, mFus_rh, pFus_lh, pFus_rh,
#     IOS_IOG_lh, pOTS_lh, IOS_IOG_pOTS_lh
"IOS_IOG_pOTS_lh" -> chosen_roi

# load significant time spans
list() -> signif_time_spans
for (contrast in c("words_minus_faces", "words_minus_cars")) {
    stringr::str_c("signif-spans-", chosen_roi, "-", contrast, ".yml") -> fname
    if (file.exists(fname)) {
        yaml::read_yaml(fname) -> signif_time_spans[contrast]
    }
}
# by observation we know there is only one significant time span from the
# clustering done in Python, so we extract it here for convenience
signif_time_spans$words_minus_cars -> temporal_roi

# Temporal ROI defined based on peak response
 temporal_roi = c(.135, .235)
# Temporal ROI defined based on spatiotemporal clustering
# temporal_roi = c(.16, .26)

# load data
readr::cols_only(subj="c",
                 intervention="c",
                 timepoint="c",
                 condition="c",
                 time="d",
                 value="d",
                 method="c",
                 roi="c") -> colspec
stringr::str_c("roi-MPM_", chosen_roi, "-timeseries-long.csv") -> fname
readr::read_csv(fname, col_types=colspec) -> rawdata

# prepare for modeling
rawdata %>%
    filter(method %in% "dSPM",
           roi %in% stringr::str_c("MPM_", chosen_roi),
           condition %in% c("words", "faces", "cars"),
           temporal_roi[1] <= time,
           temporal_roi[2] > time) %>%
    mutate(cond_=factor(condition, levels=c("words", "faces", "cars")),
           tmpt_=deviation_coding(.$timepoint, levs=c("pre", "post")),
           intv_=deviation_coding(.$intervention,
                                  levs=c("language", "letter"))) %>%
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
lme4::lmer(form, data=modeldata,
           control=lme4::lmerControl(optimizer=NULL)) -> emptymod
lme4::allFit(emptymod) -> allfit_results
print(summary(allfit_results))  # all optimizers agree on fixed effect coefs.

# show the ANOVA summary table
print(mod$anova_table)

# show the individual coefficients
print(summary(mod))

stop()

# this will give post-hoc faces-minus-words & cars-minus-words for each
# timepoint and intervention group
emmeans::emmeans(mod, "cond_", by=c("tmpt_", "intv_"), type="response",
                 contr="trt.vs.ctrl") ->
    post_hoc_conds
print(post_hoc_conds$contrasts)

# this will give post-hoc post-minus-pre for each condition in each
# intervention group
emmeans::emmeans(mod, "tmpt_", by=c("cond_", "intv_"), type="response",
                 contr="revpairwise") ->
    post_hoc_timepoints
print(post_hoc_timepoints$contrasts)

# Just examine WORD response and compare changes between intervention groups
formula(value ~ tmpt_ * intv_ + (1 | subj)) -> form
afex::mixed(form, data=filter(modeldata, cond_ == "words"), method="S",
            check_contrasts=FALSE) -> mod
print(mod$anova_table)
print(summary(mod))

# Just examine WORD response and compare for LETTER group
formula(value ~ tmpt_ + (1 | subj)) -> form
afex::mixed(form, data=filter(modeldata, cond_ == "words" & intv_ == 'letter'), method="S",
            check_contrasts=FALSE) -> mod
print(mod$anova_table)
print(summary(mod))

# Just examine WORD response and compare for LANGUAGE group
formula(value ~ tmpt_ + (1 | subj)) -> form
afex::mixed(form, data=filter(modeldata, cond_ == "words" & intv_ == 'language'), method="S",
            check_contrasts=FALSE) -> mod
print(mod$anova_table)
print(summary(mod))

# Just examine CAR response and compare for LETTER group
formula(value ~ tmpt_ + (1 | subj)) -> form
afex::mixed(form, data=filter(modeldata, cond_ == "cars" & intv_ == 'letter'), method="S",
            check_contrasts=FALSE) -> mod
print(mod$anova_table)
print(summary(mod))

# Just examine CAR response and compare for LANGUAGE group
formula(value ~ tmpt_ + (1 | subj)) -> form
afex::mixed(form, data=filter(modeldata, cond_ == "cars" & intv_ == 'language'), method="S",
            check_contrasts=FALSE) -> mod
print(mod$anova_table)
print(summary(mod))

# Just examine FACE response and compare for LETTER group
formula(value ~ tmpt_ + (1 | subj)) -> form
afex::mixed(form, data=filter(modeldata, cond_ == "faces" & intv_ == 'letter'), method="S",
            check_contrasts=FALSE) -> mod
print(mod$anova_table)
print(summary(mod))

# Just examine FACE response and compare for LANGUAGE group
formula(value ~ tmpt_ + (1 | subj)) -> form
afex::mixed(form, data=filter(modeldata, cond_ == "faces" & intv_ == 'language'), method="S",
            check_contrasts=FALSE) -> mod
print(mod$anova_table)
print(summary(mod))

# Fit model to pre data
formula(value ~ cond_ * intv_ + (1 | subj)) -> form
afex::mixed(form, data=filter(modeldata, tmpt_ == "pre"), method="S",
            check_contrasts=FALSE) -> mod
print(mod$anova_table)
print(summary(mod))

# Fit model to post data
formula(value ~ cond_ * intv_ + (1 | subj)) -> form
afex::mixed(form, data=filter(modeldata, tmpt_ == "post"), method="S",
            check_contrasts=FALSE) -> mod
print(mod$anova_table)
print(summary(mod))

# Fit model to post data for LETTER group
formula(value ~ cond_  + (1 | subj)) -> form
afex::mixed(form, data=filter(modeldata, tmpt_ == "post" & intv_ == 'letter'), method="S",
            check_contrasts=FALSE) -> mod
print(mod$anova_table)
print(summary(mod))

# # # # # # # # # # # # # # # # # # # # # # #
# Run a separate model at each time instant #
# # # # # # # # # # # # # # # # # # # # # # #

# prepare dataframe for modeling (subsetting, factor coding, etc)
rawdata %>%
    filter(method %in% "dSPM",
           roi %in% stringr::str_c("MPM_", chosen_roi),
           condition %in% c("words", "faces", "cars")) %>%
    mutate(cond_=factor(condition, levels=c("words", "faces", "cars")),
           tmpt_=deviation_coding(.$timepoint, levs=c("pre", "post")),
           intv_=deviation_coding(.$intervention,
                                  levs=c("language", "letter"))) ->
    modeldata

matrix(c(0, 1, 0, 0, 0, 1), nrow=3, ncol=2, byrow=FALSE,
       dimnames=list(c("words", "faces", "cars"), c("faces", "cars"))) ->
    contrasts(modeldata$cond_)


# sanity check: if we run a regular (non-mixed) lm on pre-subtracted data
# (post-pre and words-cars) do we get what we expect?
modeldata %>%
    tidyr::pivot_wider(id_cols=c(time, subj, cond_, intv_), names_from=tmpt_,
                       values_from=value) %>%
    mutate(post_minus_pre=post - pre) %>%
    select(-pre, -post) %>%
    tidyr::pivot_wider(id_cols=c(time, subj, intv_), names_from=cond_,
                       values_from=post_minus_pre) %>%
    mutate(words_minus_faces=words - faces,
           words_minus_cars=words - cars) %>%
    select(-words, -faces, -cars) ->
    sanitycheckdata

formula(words_minus_faces ~ intv_) -> form1
formula(words_minus_cars ~ intv_) -> form2
sanitycheckdata %>%
    nest_by(time) %>%
    mutate(words_minus_faces=list(lm(formula=form1, data=data)),
           words_minus_cars=list(lm(formula=form2, data=data))) %>%
    tidyr::pivot_longer(cols=c(words_minus_faces, words_minus_cars),
                        names_to="condition",
                        values_to="model") %>%
    rowwise() %>%
    summarise(time, condition, broom:::tidy.lm(model)) %>%
    arrange(condition) ->
    sanitycheckmod

sanitycheckmod %>%
    mutate(neglogp=-log10(p.value)) %>%
    filter(!term %in% "(Intercept)") %>%
    ggplot(mapping=aes(x=time, y=neglogp)) +
    facet_wrap(vars(condition)) +
    geom_vline(xintercept=0, col="gray60") +
    geom_line() +
    labs(y="-log₁₀(p)", x="time (s)") ->
    gg

# add horizontal reference lines corresponding to interesting p-values
list(red=0.05, blue=0.01, brown=0.001) -> pvals
for (i in seq_along(pvals)) {
    pvals[[i]] -> p
    -log10(p) -> thresh
    names(pvals)[i] -> color
    gg +
        geom_hline(yintercept=thresh, linetype="dotted", col=color) +
        annotate("text", x=1, y=thresh, label=stringr::str_c("p=", p), hjust=1,
                 vjust=-0.5, col=color, size=6/.pt) ->
        gg

}
ggsave(filename="sanity-check-pvals.png", plot=gg)


# # actually run the full mixed models
# formula(value ~ cond_ * tmpt_ * intv_ + (1 + cond_ + tmpt_ | subj)) -> form
# modeldata %>%
#     nest_by(time) %>%
#     mutate(model=list(afex::lmer_alt(formula=form, data=data, method="S",
#                                      check_contrasts=FALSE))) %>%
#     summarise(broom.mixed::tidy(model)) %>%
#     tidyr::unnest(cols=c()) ->
#     mod
# readr::write_csv(mod, file="full-model-results.csv")
#
# # reduce the dataframe to just coef and p-value of fixed effects
# mod %>%
#     filter(effect == "fixed") %>%
#     select(time, term, estimate, statistic, df, p.value) %>%
#     rename(t=statistic, p=p.value, coef=estimate) %>%
#     mutate(neglogp=-log10(p)) ->
#     mod_short
# readr::write_csv(mod_short, file="short-model-results.csv")
#
# # run one model at t=0 in case we need to inspect the model object
# afex::lmer_alt(formula=form, data=filter(modeldata, time == 0.0), method="S",
#                check_contrasts=FALSE) ->
#     one_model


# # # # # # #
# PLOTTING  #
# # # # # # #

# # make faceted plots for each fixed effect, of the p-values at each time instant
# mod_short %>%
#     filter(!term %in% "(Intercept)") %>%
#     ggplot(mapping=aes(x=time, y=neglogp)) +
#     facet_wrap(vars(term)) +
#     geom_vline(xintercept=0, col="gray60") +
#     geom_line() +
#     labs(y="-log₁₀(p)", x="time (s)") ->
#     gg
#
# # add horizontal reference lines corresponding to interesting p-values
# list(red=0.05, blue=0.01, brown=0.001) -> pvals
# for (i in seq_along(pvals)) {
#     pvals[[i]] -> p
#     -log10(p) -> thresh
#     names(pvals)[i] -> color
#     gg +
#         geom_hline(yintercept=thresh, linetype="dotted", col=color) +
#         annotate("text", x=1, y=thresh, label=stringr::str_c("p=", p), hjust=1,
#                  vjust=-0.5, col=color, size=6/.pt) ->
#         gg
#
# }
#
# ggsave(filename="mixmod-by-timepoint-pvals.png", plot=gg)
