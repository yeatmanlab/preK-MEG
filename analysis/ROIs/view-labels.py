# uncomment and do this first if you're in ipython or a notebook:
# %gui qt

import mne
from surfer import Brain

# load hand-drawn FreeView label
lab = mne.read_label('VOTC-lh.label')

# label names from 10.1016/j.neuroimage.2010.06.010

label_names = ('G_and_S_occipital_inf',      # 2
               'G_oc-temp_lat-fusifor',      # 21
               'G_temporal_inf',             # 37
               'S_collat_transv_ant',        # 50
               'S_collat_transv_post',       # 51
               'S_occipital_ant',            # 59
               'S_oc-temp_lat',              # 60
               'S_oc-temp_med_and_Lingual',  # 61
               'S_temporal_inf',             # 72
               )

regexp = '|'.join(label_names)
labels = mne.read_labels_from_annot(subject='fsaverage', parc='aparc.a2009s',
                                    hemi='lh', subjects_dir=None,
                                    regexp=regexp)


# FSAverage w/ hand-drawn label (magenta) and atlas labels (transparent yellow)
brain = Brain('fsaverage', 'lh', 'inflated')
brain.add_label(lab, color='m', alpha=1)
for l in labels:
    brain.add_label(l, alpha=0.5, color='y')


# possible ones to add?
addl_label_names = ('G_occipital_middle',     # 19
                    'G_oc-temp_med-Lingual',  # 22
                    )

regexp = '|'.join(addl_label_names)
addl_labels = mne.read_labels_from_annot(subject='fsaverage',
                                         parc='aparc.a2009s',
                                         hemi='lh', subjects_dir=None,
                                         regexp=regexp)

for l in addl_labels:
    brain.add_label(l, alpha=0.5, color='c')
