#!/usr/bin/env python

"""Gist to fit XDAWN filter to ASSR response and viz ERF topography."""

__author__ = "Kambiz Tavabi"
__copyright__ = "Copyright 2019, Seattle, Washington"
__credits__ = ["Eric Larson"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Kambiz Tavabi"
__email__ = "ktavabi@uw.edu"
__status__ = "Production"

import mne
datapath = '/media/ktavabi/ALAYA/data/ilabs/badbaby/'
event_id = {'Auditory': 1}
raw_fname = datapath + 'larson_eric_tone_raw.fif'
tmin, tmax = -0.5, 2.


#   Setup for reading the raw data
def preproc_raw(fname, h_freq=50.):
    this_raw = mne.io.read_raw_fif(fname, allow_maxshield='yes')
    mne.channels.fix_mag_coil_types(this_raw.info)
    this_raw = mne.preprocessing.maxwell_filter(this_raw)
    this_raw.filter(None, h_freq, fir_design='firwin', n_jobs='cuda')
    return this_raw


# ASSR
print('Loading ASSR')
raw = preproc_raw(raw_fname)
events = mne.find_events(raw)
picks = mne.pick_types(raw.info, meg=True)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13),
                    preload=True)
#epochs.average().plot()

# signal_cov = mne.compute_covariance(epochs, tmin=0, tmax=1.5)
# signal_cov = mne.cov.regularize(signal_cov, raw.info)
signal_cov = mne.compute_covariance(epochs, method='oas', n_jobs=18)
signal_cov = mne.cov.regularize(signal_cov, epochs.info, rank='full')

print('Fitting Xdawn')
xd = mne.preprocessing.Xdawn(n_components=1, signal_cov=signal_cov,
                             correct_overlap=False, reg='ledoit_wolf')
xd.fit(epochs)

#xd.apply(epochs)['Auditory'].average().plot_topo()

tEI = epochs.info;
tEI['sfreq']=1; # we want each second to represent a different row of filter matrix

#mne.EvokedArray( xd.filters_['Auditory'], tEI ).plot_topomap(times=[0,1,2,3,4]);
#figure(); plot( epochs.times, np.dot( xd.filters_['Auditory'][[1,2,3,4,0],:], epochs.average().data ).T );
## figure editor can be used to assign contrasting color and linewidth to last trace

tNoi = np.dot( xd.filters_['Auditory'][1:,:].mean(0), epochs.average().data ).T;
tSig = np.dot( xd.filters_['Auditory'][0,:], epochs.average().data ).T;
tSNTopo = np.array( [ xd.filters_['Auditory'][:,0].T, xd.filters_['Auditory'][:,1:].mean(1).T ] ).T;
mne.EvokedArray( tSNTopo, tEI ).plot_topomap(times=[0,1]);
figure(); plot( epochs.times, np.array([ tSig, tNoi ]).T ); legend( [ 'signal', 'noise' ] );

