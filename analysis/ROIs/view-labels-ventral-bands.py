# uncomment and do this first if you're in ipython or a notebook:
# %gui qt

import mne
from surfer import Brain

##############################################
# PROGRESSIVE ROIs ALONG THE VENTRAL SURFACE #
##############################################

# colors for the 5 ROIs (index 0 is not used)
colors = list('-rgbcm')

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
regions = {1: [f'lateraloccipital_{n}-lh' for n in (4, 5)] + ['lingual_2-lh'],
           2: [f'lateraloccipital_{n}-lh' for n in (3, 6, 11)] +
              ['lingual_1-lh'],
           3: [f'lateraloccipital_{n}-lh' for n in (7, 10)] +
              ['fusiform_2-lh', 'lingual_4-lh'],
           4: ['lateraloccipital_9-lh', 'inferiortemporal_8-lh',
               'lingual_8-lh'] +
              [f'fusiform_{n}-lh' for n in (1, 3)],
           5: [f'fusiform_{n}-lh' for n in (4, 5)] +
              [f'inferiortemporal_{n}-lh' for n in (5, 6, 7)],
           }
kwargs = dict(subject='fsaverage', parc='aparc_sub', hemi='lh',
              subjects_dir=None)
for r, label_names in regions.items():
    color = colors[r]
    regexp = r'|'.join(label_names)
    regions[r] = mne.read_labels_from_annot(regexp=regexp, **kwargs)


# do the drawing
brain = Brain('fsaverage', 'lh', 'inflated', views='ventral')
# aparc_sub
for reg, lablist in regions.items():
    color = colors[reg]
    for label in lablist:
        brain.add_label(label, color=color)
# hand-drawn
for reg, lab in ventral_bands.items():
    color = colors[reg]
    brain.add_label(lab, color=color, alpha=0.5)


# code for playing around with regions to see which goes in which band
aparc_labels = mne.read_labels_from_annot(
    subject='fsaverage', parc='aparc_sub', hemi='lh', subjects_dir=None,
    regexp=r'lateraloccipital|lingual|fusiform|inferiortemporal')
n = 0
brain.add_label(aparc_labels[n], color='y')
brain.remove_labels(aparc_labels[n].name)
