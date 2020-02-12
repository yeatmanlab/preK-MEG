# uncomment and do this first if you're in ipython or a notebook:
# %gui qt

import mne
from surfer import Brain
from matplotlib.colors import to_rgba

##############################################
# PROGRESSIVE ROIs ALONG THE VENTRAL SURFACE #
##############################################

# colors for the 5 ROIs (index 0 is not used)
roi_colors = list('-rgbcm')  # index 0 won't be used

# load hand-drawn labels
ventral_bands = {1: 'VOTC-1-lh.label',
                 2: 'VOTC-2-lh.label',
                 3: 'VOTC-3-lh.label',
                 4: 'VOTC-4-lh.label',
                 5: 'VOTC-5-lh.label',
                 }
for label_number, label_name in ventral_bands.items():
    ventral_bands[label_number] = mne.read_label(label_name)

# load aparc_sub labels
rois = dict()
for region_number in range(1, 6):
    label = mne.read_label(f'ventral_band_{region_number}-lh.label')
    label.subject = 'fsaverage'
    label.color = to_rgba(roi_colors[region_number])
    rois[region_number] = label


# do the drawing
brain = Brain('fsaverage', 'lh', 'inflated', views='ventral')
# aparc_sub
for reg, label in rois.items():
    brain.add_label(label)
# hand-drawn
for reg, label in ventral_bands.items():
    brain.add_label(label, color=roi_colors[reg], alpha=0.5)


# # code for playing around with regions to see which goes in which band
# aparc_labels = mne.read_labels_from_annot(
#     subject='fsaverage', parc='aparc_sub', hemi='lh', subjects_dir=None,
#     regexp=r'lateraloccipital|lingual|fusiform|inferiortemporal')
# n = 0
# brain.add_label(aparc_labels[n], color='y')
# brain.remove_labels(aparc_labels[n].name)
