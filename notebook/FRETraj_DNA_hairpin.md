---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# FRET-assisted modeling of a DNA hairpin

This notebook describes the step in analyzing an molecular dynamics (MD) simulation of a biomolecule with *FRETraj*. 

First, we compute multiple **accessible-contact volumes** (ACV) along the MD trajectory and calculate a distance distribution between donor and acceptor fluorophores. Based on the inter-dye distances we then compute photon emission events which ultimately gives us shot-noise limited FRET histograms that can be compared to experiments.

```python
import numpy as np
import mdtraj as md
import fretraj as ft
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style('white')
sns.set_context('notebook')
sns.set(font='Arial')

def set_ticksStyle(x_size=4, y_size=4, x_dir='in', y_dir='in'):
    sns.set_style('ticks', {'xtick.major.size': x_size, 'ytick.major.size': y_size, 'xtick.direction': x_dir, 'ytick.direction': y_dir})
```

## Loading the MD trajectory and defining dye parameters


Start by loading an **MD trajectory** (.xtc) with an associated **topology** file (.pdb)

```python
traj = md.load('../data/DNA_hairpin.xtc', top='../data/DNA_hairpin.pdb')
```

Inspect the trajectory's length and timestep

```python
print(f'timestep: {traj.timestep/1000 :.1f} ns')
print(f'length: {traj.time[-1]/1000 :.0f} ns')
```

A snapshot of the DNA hairpin structure is save every 100 **picosecond**. This timestep is reasonable considering the fluorescence lifetime of the donor (~1 ns). Rotational diffusion of the fluorophores, determined experimentally by dynamic fluorescence anisotropy decays, is fast compared to the fluorescence lifetime while conformational dynamics of the biomolecule are expected to be slower. 

We can inspect the flexibility of the DNA hairpin directly with NGLview.

```python
ft.jupyter.nglview_trajectory(traj)
```

The FRET construct is designed to give a mean FRET efficiency around the Förster radius of Cy3-Cy5 ($R_0=5.4$ nm) for maximal sensitivity. The DNA is labeled at the internal **T20** (on C7) and at the 3'-terminal **C20** (on O5\').
<img src="../data/secondary_structure.png" width=200>
Lets find the **IDs** (serials) of the attachment positions (the serials are 1-based)

```python
traj_df = traj.top.to_dataframe()[0]
traj_df.loc[((traj_df['resSeq']==20) & (traj_df['name']=='C7')) | ((traj_df['resSeq']==44) & (traj_df['name']=='P1'))]
```

Knowing the atom identifiers of the attachment sites we can define the **dye-linker parameters** for the **ACV** simulations.

```python
labels = {"Position":
            {"Cy3-20-C5":
                {"attach_id": 626,
                 "linker_length": 20,
                 "linker_width": 5,
                 "dye_radius1": 8,
                 "dye_radius2": 3,
                 "dye_radius3": 1.5,
                 "cv_fraction": 0.57,
                 "cv_thickness": 3,
                 "use_LabelLib": True},
             "Cy5-44-P1":
                {"attach_id": 1395,
                 "linker_length": 20,
                 "linker_width": 5,
                 "dye_radius1": 9.5,
                 "dye_radius2": 3,
                 "dye_radius3": 1.5,
                 "cv_fraction": 0.39,
                 "cv_thickness": 3,
                 "use_LabelLib": True},
            },
         "Distance": {"Cy3-Cy5":
            {"R0": 54}
            }
         }
```

Check the compulsory parameters for completeness. The parameters can be saved to disk if desired.

```python
ft.cloud.check_labels(labels, verbose=False)
#ft.cloud.save_labels('DNA_hairpin_ACV_parameters.json', labels)
```

## Computing accessible-contact volumes
Calculate donor and acceptor **accessible contact volumes (ACV)** for a few selected frames (here snapshots spaced 100 ns apart)

Please note: we don't compute ACVs for all 10000 frames just yet because keeping all ACV in memory at the same time is unpractical on consumer hardware. See [below](#Scaling-ACV-simulation-and-FRET-prediction) how we tackle this issue.

```python
selected_frames = range(0,10001, 1000)
acv_D = ft.cloud.Volume.from_frames(traj, 'Cy3-20-C5', labels, selected_frames)
acv_A = ft.cloud.Volume.from_frames(traj, 'Cy5-44-P1', labels, selected_frames)
traj_sliced = traj.slice(selected_frames)
```

We get a list of donor and acceptor ACVs (as `fretraj.cloud.Volume` ojects). To visualize them on the structure we first convert them to `mdtraj.Trajectory` objects.

```python
acv_D_traj = ft.cloud.create_acv_traj(acv_D)
acv_A_traj = ft.cloud.create_acv_traj(acv_A)
```

Now, we can visualize them

```python
ft.jupyter.nglview_trajectory_ACV(traj_sliced, acv_D_traj['FV'], acv_A_traj['FV'],
                                  acv_D_traj['CV'], acv_A_traj['CV'])
```

## Scaling ACV simulation and FRET prediction

For memory-efficient FRET predictions across the entire MD trajectory we compute donor and acceptor ACVs on-the-fly and calculate a mean inter-dye distance $R_{DA}$ and FRET efficiency $E_{DA}$ before moving to the next frame. The memory-intensive ACVs are not thus stored.

```python
# test with a few frames (snapshot every 100 ns)
fret = ft.cloud.pipeline_frames(traj, 'Cy3-20-C5', 'Cy5-44-P1', labels, selected_frames, 'Cy3-Cy5')

# full trajectory (snapshot every 100 ps, time-consuming!)
#all_frames = range(0,10001)
#fret = ft.cloud.pipeline_frames(traj, 'Cy3-20-C5', 'Cy5-44-P1', labels, all_frames, 'Cy3-Cy5')
```

The resulting list of `fretraj.cloud.FRET` objects can be serialized ("pickled") and saved to disk to avoid recalculation.

```python
#ft.cloud.save_obj('../data/FRET_DNA_hairpin.pkl', fret)
fret = ft.cloud.load_obj('../data/FRET_DNA_hairpin.pkl')
```

Next, we generate a `fretraj.cloud.Trajectory` object from the FRET list.

```python
fret_traj = ft.cloud.Trajectory(fret, timestep=traj.timestep, kappasquare=0.66)
fret_traj.dataframe.head()
```

We can plot various **distance metrics** and the **FRET distribution**. The width of this distribution reflects the conformational dynamics within the simulation time of 1 $\mu$s.

```python
with sns.axes_style('ticks'):
    set_ticksStyle()
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(5.5, 1.75), sharex=False, sharey=True, squeeze=False)    
    ax[0,0].hist(fret_traj.R_attach, bins=25, range=[37, 65], color='gray', edgecolor='w', linewidth=0.5, zorder=0)
    ax[0,0].hist(fret_traj.mean_R_DA, bins=25, range=[37, 65], color=[0.23,0.37,0.64], edgecolor='w', linewidth=0.5, zorder=0)
    ax[0,1].hist(fret_traj.mean_E_DA, bins=25, range=[0,1], color=[0.75,0.51,0.38], edgecolor='w', linewidth=0.5, zorder=0)
    ax[0,0].set_xlabel('distance ($\mathregular{\AA}$)')
    ax[0,1].set_xlabel('FRET')
    ax[0,0].set_ylabel('occurence')
    ax[0,0].legend(['$\mathregular{{\it R}_{attach}}$', '$\mathregular{{\it R}_{DA}}$'], frameon=False, handlelength=0.75)
    ax[0,1].legend(['$\mathregular{{\it E}_{DA}}$'], frameon=False, handlelength=0.75)
```

## Adding shot noise by simulating photon emission

To compare the FRET predictions with single-moelcule eperiments we need to take the photon statistics into account. For this purpose, we generate photon emission events based on the experimental burst size distribution and the time-dependent inter-dye distances $R_{DA}$.

We first save a time trace of $R_{DA}$ and $\kappa^2$ values (the latter is set to 0.66, i.e. isotropic average).

```python
fret_traj.save_traj('../data/R_kappa.dat', format='txt', R_kappa_only=True, units='nm', header=False)
```

```python
with sns.axes_style('ticks'):
    set_ticksStyle()
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.5, 1.75), sharex=False, sharey=True, squeeze=False)    
    ax[0,0].plot(fret_traj.dataframe['time (ps)']/1000, fret_traj.mean_R_DA, 'k')
    ax[0,0].set_ylim(37, 65)
    ax[0,0].set_xlabel('time (ns)')
    ax[0,0].set_ylabel('$\mathregular{{\it R}_{DA}~(\AA)}$')
```

We need to define parameters for the burst simulations

```python
parameters = {'dyes': {
                  'tauD': 1,
                  'tauA': 1.4,
                  'QD': 0.2,
                  'QA': 0.2,
                  'dipole_angle_abs_em': 10},
              'sampling': {
                  'nbursts': 20000,
                  'skipframesatstart': 0,
                  'skipframesatend': 1000,
                  'multiprocessing': True},
              'fret': {
                  'R0': 5.4,
                  'kappasquare': 0.666666,
                  'no_gamma': False,
                  'quenching_radius': 1.0},
              'species': {
                  'name': ['all'],
                  'unix_pattern_rkappa': ['../data/R_kappa.dat'],
                  'probability': [1],
                  'n_trajectory_splits': None},
              'bursts': {
                  'lower_limit': 50,
                  'upper_limit': 150,
                  'lambda': -2.3,
                  'burst_size_file': '../data/burst_sizes.dat',
                  'QY_correction': False,
                  'averaging': 'all'}}
```

We can then initialize a burst experiment

```python
experiment = ft.burst.Experiment('.', parameters, compute_anisotropy=False)
```

The resulting `fretraj.burst.Experiment` objects can be serialized and saved to disk.

```python
#experiment.save('../data/DNA_hairpin_fret_burst.pkl', remove_bursts=True)
```

Comparing the FRET histogram before and after photon simulation shows how shot-noise broadens the FRET distribution.

```python
with sns.axes_style('ticks'):
    set_ticksStyle()
    f, ax = plt.subplots(nrows=2, ncols=1, figsize=(3, 3), sharex=True, sharey=True, squeeze=False)    
    ax[0,0].hist(fret_traj.mean_E_DA, bins=25, range=[0,1], color=[0.75,0.51,0.38], edgecolor='w', linewidth=0.5, zorder=0)
    ax[1,0].hist(experiment.FRETefficiencies, bins=25, range=[0,1], color=[0.75,0.51,0.38])
    ax[0,0].legend(['conf.\ndynamics'], frameon=False, handlelength=0, loc='upper left')
    ax[1,0].legend(['shot-noise\nbroadening'], frameon=False, handlelength=0)
    ax[1,0].set_xlabel('FRET')
    ax[0,0].set_ylabel('occurence')
    ax[1,0].set_ylabel('occurence')
```

```python

```
