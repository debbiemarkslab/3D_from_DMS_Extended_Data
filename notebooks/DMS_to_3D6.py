from evcouplings.compare import DistanceMap, PDB, ClassicPDB
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde
from scipy.stats import hypergeom
from collections import Counter
from ast import literal_eval
from copy import deepcopy
from scipy import stats

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib as mlib
import seaborn as sns
import pandas as pd
import numpy as np


# Importing datasets ##########################################################

def load_sheet(excel_doc, sheet):
    '''loads in mutant sheets, skipping header, clearing empty cols'''

    table = excel_doc.parse(sheet, skiprows=1)
    table = table.loc[:, [c for c in table.columns if ('Unnamed' not in c)]]
    
    table.loc[:, 'mut'] = table['mut'].apply(lambda x: x.split(','))
    table.loc[:, 'positions'] = table['positions'].apply(eval)
    
    for p, k in enumerate(['i', 'j']):
        table.loc[:, k] = table['positions'].apply(lambda x: x[p])
        table.loc[:, 'A_'+k] = table['mut'].apply(lambda x: x[p][0])
        table.loc[:, 'mut.'+k] = table['mut'].apply(lambda x: x[p])

    if 'lnWs' in table.columns:
        table.loc[:, 'lnWs'] = table['lnWs'].apply(lambda x: eval(x.replace('nan', 'np.nan')))
    
    return(table)


# Importing 3D structures #####################################################


heavy_atoms = [
    'OXT','OG1','CD2','CZ2','CA','SD','CE2','OD1','N','O',
    'OE1','OD2','NE1','CE','NE2','CZ3','CG1','CZ','CB',
    'CG2','CD','CG','CE3','NZ','CE1','CD1','OE2','ND2','C',
    "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'",
    "C1'", 'N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6',
    'P', 'OP1', 'OP2', 'N9', 'C8', 'N7', 'O6', 'N2', 'O4',
    'N6', 'MG', 'O'
]


def get_xtal(pdb_id, pdb_chain, offset=0):
    '''loads in pdb data for pdb id, chain- adjusts index by offset'''

    if '.pdb' in pdb_id:
        coords = ClassicPDB.from_file(pdb_id).get_chain(pdb_chain)
    else:
        coords = PDB.from_id(pdb_id).get_chain(pdb_chain)

    dmap = DistanceMap.from_coords(coords.filter_atoms(heavy_atoms))
    dists = dmap.contacts(500)

    dists.loc[:, 'positions'] = dists.apply(
        lambda x: (int(x['i'])+offset, int(x['j'])+offset), axis=1)
    
    seq = dmap.residues_i.set_index('coord_id')['one_letter_code']
    dists.loc[:, 'A_i'] = dists['i'].apply(lambda x: seq.loc[x])
    dists.loc[:, 'A_j'] = dists['j'].apply(lambda x: seq.loc[x])
    
    return(dists)


def get_dists(pdbs, dms):
    '''gathers distances for a list of pdbs-- adds dists to dms table'''

    xtals = {}
    for x, c, o in pdbs:
        cmap = get_xtal(x, c, offset=o).set_index('positions')
        xtals[x] = cmap['dist']
        dists = dict(cmap['dist'])
        dms.loc[:, 'dist.'+x] = dms['positions'].apply(
            lambda x: dists[x] if x in dists else np.nan)
    
    xtals = pd.DataFrame(xtals)
    
    # also include min distance between pairs in any structure
    xtals.loc[:, 'min_overall'] = xtals.apply(lambda x: np.min(x), axis=1)
    min_dist = dict(xtals['min_overall'])
    dms.loc[:, 'dist.any_struct'] = dms['positions'].apply(
        lambda x: min_dist[x] if x in min_dist else np.nan)

    return(xtals.reset_index())


def get_ss(pdb_id, pdb_chain, offset=0):
    '''loads in pdb data for pdb id, chain- adjusts index by offset'''
    if '.pdb' in pdb_id:
        coords = ClassicPDB.from_file(pdb_id).get_chain(pdb_chain)
    else:
        coords = PDB.from_id(pdb_id).get_chain(pdb_chain)
    res = coords.residues
    res.loc[:, 'id'] = res['id'].astype(int) + offset
    return(res.loc[:,['id', 'sec_struct_3state']])


def merge_ss(pdbs):
    '''extracting the union of multiple secondary structures--
        assuming that alpha helix and beta strand don't overlap,
        if they do, which they don't for these proteins, beta gets priority'''
    m = get_ss(*pdbs[0])
    for pdb in pdbs[1:]:
        s = get_ss(*pdb).rename(columns={'sec_struct_3state': pdb[0]})
        m = pd.merge(m, s, on='id', how='outer')
    
    m.loc[:, 'joint'] = m.apply(
        lambda x: 'E' if any([a=='E' for a in x]) else ('H' if any([a=='H' for a in x]) else 'C'),
    axis=1)
    
    return(m)

# Fitness distributions and projection ########################################


def plot_double_fitness_nogrid(dms, fit='lnW', proj='lnW.proj', fits='lnWs', epi='epi',
                       mi='mut.i', mj='mut.j', outfile=None):
    '''plot double mutant fitness VS projection,
    and distributions of double/single mutant fitnesses'''
    fig = plt.figure(figsize=(10, 5))
    axs = [fig.add_subplot(x) for x in [(321), (323), (122), (325)]]
    singles_i = dms.drop_duplicates(mi).set_index(mi)
    singles_i = singles_i[fits].apply(lambda x: x[0])
    singles_j = dms.drop_duplicates(mj).set_index(mj)
    singles_j = singles_j[fits].apply(lambda x: x[1])
    singles = pd.concat([singles_i, singles_j]).reset_index().drop_duplicates('index')

    print(singles.head(), singles_i.head())
    axs[0].hist(singles[fits].dropna(), bins=20)
    axs[1].hist(dms[fit].dropna(), bins=20)

    measd = ~dms[proj].isnull() & ~dms[fit].isnull()
    axs[3].hist(dms.loc[measd, epi], bins=20)

    scatter_density(dms.loc[measd, proj], dms.loc[measd, fit], axs[2])
    x = [np.nanmin(dms[proj]), np.nanmax(dms[proj])]
    axs[2].plot(x, x,'--k', lw=2)
    d = dms[~dms[proj].isnull() & ~dms[fit].isnull()]
    grad, interc, r_val, p_val, std = stats.linregress(d[proj], d[fit])
    axs[2].legend(['model: r value = ' + str(r_val)[:5],
                   'measured'])

    axs[0].set_xlabel('single mutant fitness')
    axs[0].set_ylabel('# of single mutants')
    axs[1].set_xlabel('double mutant fitness')
    axs[1].set_ylabel('# of double mutants')
    axs[2].set_xlabel('projected double mutant fitness')
    axs[2].set_ylabel('double mutant fitness')
    axs[3].set_xlabel('double mutant epistasis')
    axs[3].set_ylabel('# of double mutants')
    fig.tight_layout()

    for a in axs:
        a.grid(False)
        a.tick_params(direction='out', length=4, width=1.5, axis='y')
        a.tick_params(direction='out', length=4, width=1.5, axis='x')

    if not outfile is None:
        fig.savefig(outfile, dpi=300)


def scatter_density(x, y, ax):
    '''creates scatter plot, coloring dots by density in area'''
    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    ax.scatter(x, y, c=z, s=50, edgecolor='', cmap='viridis')


# Mutant library coverage #####################################################


def plot_mutant_coverage_nogrid(dms, rng, vms=[0, 361], outfile=None, n=10):
    '''represent coverage and bias in double mutant'''
    fig = plt.figure(figsize=(10, 5))
    axs = [fig.add_subplot(x) for x in [(121), (222), (224)]]

    map_ij_coverage(dms, rng, axs[0], vms=vms, n=n)
    hist_ij_coverage(dms, axs[1])
    hist_i_coverage(dms, axs[2])

    axs[0].set_title('# of mutants at i,j')
    axs[1].set_xlabel('# of mutants at i,j')
    axs[1].set_ylabel('# of pairs (i,j)')
    axs[2].set_xlabel('# of mutants at i')
    axs[2].set_ylabel('# of sites (i)')
    fig.tight_layout()

    for a in axs:
        a.grid(False)
        a.tick_params(direction='out', length=4, width=1.5, axis='y')
        a.tick_params(direction='out', length=4, width=1.5, axis='x')
    
    if not outfile is None:
        fig.savefig(outfile, dpi=300)


def map_ij_coverage(tbl, rng, ax, vms=[0, 361], n=10):
    '''plots the number of double mutants'''
    covg = dict(Counter(tbl['positions']))
    hmap = np.zeros((len(rng), len(rng)))
    hmap[:][:] = np.nan
    offset = np.min(rng)
    for k, i in enumerate(rng):
        for j in rng[k+1:]:
            if (i, j) in covg:
                hmap[i-offset][j-offset] = covg[(i, j)]
                hmap[j-offset][i-offset] = covg[(i, j)]

    cmap = mlib.cm.Blues
    cmap.set_bad('gray')
    cm = ax.imshow(hmap, cmap=cmap, vmin=vms[0], vmax=vms[1])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(cm, cax=cax)
    ax.set_xticks((rng - np.min(rng))[::n])
    ax.set_xticklabels(rng[::n])
    ax.set_yticks((rng - np.min(rng))[::n])
    ax.set_yticklabels(rng[::n])

    return(hmap)


def hist_ij_coverage(tbl, ax):
    '''plot distribution of number of doubles sampling each i,j'''
    ax.hist(tbl.groupby('positions')['mut'].apply(len), bins=20)


def hist_i_coverage(tbl, ax):
    '''plot distribution of number of doubles sampling each i'''
    tbl_i = tbl.groupby('i')['mut'].apply(len).reset_index()
    tbl_j = tbl.groupby('j')['mut'].apply(len).reset_index()
    tbl_sites = pd.merge(tbl_i, tbl_j.rename(columns={'j':'i'}), on='i', how='outer')
    tbl_sites.loc[:, 'count'] = tbl_sites.apply(lambda x: np.nansum([x['mut_x'],x['mut_y']]), axis=1)
    ax.hist(tbl_sites['count'], bins=20)


# Selecting and sorting the top most epistatic mutants at each residue pair ###


def most_epistatic_pairs(epi_table, epi='epi', sign='positive'):
    '''finds the double mutant with highest epistasis at each i-j pair'''

    top_pairs = epi_table.sort_values(epi, ascending=(sign != 'positive'))
    top_pairs = top_pairs.drop_duplicates('positions', keep='first') 
    num_pairs = epi_table.groupby('positions')['mut'].apply(len)
    num_pairs = num_pairs.reset_index().rename(columns = {'mut': 'num.muts'})
    
    top_pairs = pd.merge(top_pairs, num_pairs, on='positions')
    
    return(top_pairs)

# epistasis visualizations ################################################


def epi_map(top_epi, lims, vms=[None, None],
            e='epi', n=10, buff=2, tixm=None, tixM=None):
    '''represent epistasis throughout i,j pairs as heatmap'''
    offset = lims[0]
    L = lims[1] - lims[0] + 1
    covg = set(top_epi['positions'])
    epi = top_epi.set_index('positions')[e]
    hmap = np.zeros((L, L))
    hmap[:][:] = np.nan
    for (i, j) in covg:
        hmap[i-offset][j-offset] = epi[(i, j)]
        hmap[j-offset][i-offset] = epi[(i, j)]

    cmap = mlib.cm.Blues
    cmap.set_bad('white')
    fig, ax = plt.subplots(1)
    ax.imshow(hmap, cmap=cmap, vmin=vms[0], vmax=vms[1])

    # formatting niceties:
    if tixm is None:
        tixm = lims
    if tixM is None:
        tixM = lims

    tixm = np.arange(*tixm, 1)
    tixM = np.arange(*tixM, n)

    ax.set_xticks(tixm-offset, minor=True)
    ax.set_yticks(tixm-offset, minor=True)
    ax.set_xticks(tixM-offset, minor=False)
    ax.set_yticks(tixM-offset, minor=False)
    ax.set_xticklabels(tixM, fontsize=26)
    ax.set_yticklabels(tixM, fontsize=26)
    ax.set_xlim(-buff, L+buff)
    ax.set_ylim(-buff, L+buff)
    ax.invert_yaxis()

    return(fig, ax, hmap)


# predicted contact visualizations ##########################################

def plotIJ(df, ax=None, color='black', s=100, offset=0, alpha=1, z=None):
    '''scatter plot i,j pairs provided'''
    if ax is None:
        fig, ax = plt.subplots(1)
        fig.set_size_inches(10, 10)
        ax.invert_yaxis()

    i = df['positions'].apply(lambda x: x[0] + offset)
    j = df['positions'].apply(lambda x: x[1] + offset)

    ax.scatter(i, j, color=color, s=s, alpha=alpha,
               edgecolors='none', zorder=z)
    ax.scatter(j, i, color=color, s=s, alpha=alpha,
               edgecolors='none', zorder=z)


def in_lim(pos, lims):
    '''report if position falls within provided range'''
    i_in = np.all([p >= lims[0] for p in pos])
    j_in = np.all([p <= lims[1] for p in pos])
    return(i_in & j_in)


def plotContacts(df, ax=None, s=300, offset=0, dist='min_overall',
                 c1='#e3eef8', c2='#96c4de', lims=None, tixm=None, tixM=None,
                 buff=2, alpha=1, d1=8, d2=5, n=10, label_offset=0):
    '''scatter plot i,j pairs within 8A, and 5A'''

    if ax is None:
        fig, ax = plt.subplots(1)
        fig.set_size_inches(10,10)
    else:
        fig = None

    if lims is None:
        x = df['positions'].apply(lambda x: x[0])
        lims = [np.min(x), np.max(x)]

    inlim = df['positions'].apply(lambda x: in_lim(x, lims))

    plotIJ(df[df[dist].lt(d1) & inlim],
           color=c1, s=s, alpha=alpha, ax=ax, z=0)
    plotIJ(df[df[dist].lt(d2) & inlim],
           color=c2, s=s, alpha=alpha, ax=ax, z=1)

    # formatting niceties:
    if tixm is None:
        tixm = lims
    if tixM is None:
        tixM = lims
    tixm = np.arange(*tixm, 1)
    tixM = np.arange(*tixM, n)

    ax.set_xticks(tixm, minor=True)
    ax.set_yticks(tixm, minor=True)
    ax.set_xticks(tixM, minor=False)
    ax.set_yticks(tixM, minor=False)
    ax.set_xticklabels(tixM+label_offset, fontsize=26)
    ax.set_yticklabels(tixM+label_offset, fontsize=26)
    ax.set_xlim(lims[0]-buff, lims[1]+buff)
    ax.set_ylim(lims[0]-buff, lims[1]+buff)
    ax.invert_yaxis()

    return(fig, ax)


def compare_contacts_verbose(X, Y, Z, d, D, name, Ns,
                     lims, tixM, tixm, tN=10, o=0, oL=0, s=300):#, Lo=0):
    '''plot epi pairs versus crystal structures'''
    M = len(set(Z['positions']))
    n = np.sum(Z.drop_duplicates('positions')['dist.any_struct'].lt(5))

    for N in Ns:
        plt.close()
        fig, ax = plotContacts(X, lims=lims, dist=d, tixM=tixM, tixm=tixm,
                               n=tN, offset=o, s=s, label_offset=oL)
        plotIJ(Y.head(N), ax=ax, z=2, s=100, offset=o)
        ax.tick_params(width=4, length=16, which='major')
        ax.tick_params(width=2, length=9, which='minor')
        hits = np.sum(Y.head(N)[D].lt(5))
        print(hits-1, M, n, N)
        pv = hypergeom.sf(hits-1, M, n, N)
        pv = '%.2E' % pv

        ax.set_title('top '+str(N)+' epistasis pairs ('+str(hits) +
                     ' true contacts; p-value: '+pv+')', fontsize=20)

        #fig.savefig('../supplementary_figures/epi_contacts/'+name +
        #            '/'+name+'--top_'+str(N)+'_epi_pairs.svg')

# secondary structure score ##############################################


def compute_SS(epi, fill=np.nan, aparams=[1, 0.723, 0.635, 0.587], bparams=[1, 0.723],
               e='epi', l=1, r=1, bcut=0.75, acut=1.5, nb=2, na=4):
    '''computes SS scores according to Perry's algorithm'''

    epi_i = epi.sort_values('i').drop_duplicates('i')
    epi_ij = epi.set_index(['i', 'j'])[e].astype(float)
    pairs = set(epi_ij.index)
    indices = list(epi_i['i'])

    # complete values where data is missing
    for i in indices:
        for j in np.arange(i+1, i+5):
            if (i, j) not in pairs:
                epi_ij.loc[(i, j)] = fill
        for j in np.arange(i-4, i):
            if (j, i) not in pairs:
                epi_ij.loc[(j, i)] = fill

    std_i1 = np.nanvar(epi_ij.loc[[(i, i+1) for i in indices[:-1]]])

    alpha = []
    beta = []

    for i in indices:
        alpha.append(alpha_score(epi_ij, i, std_i1, aparams, l, r))
        beta.append(beta_score(epi_ij, i, std_i1, bparams, l, r))

    scores = pd.DataFrame({
        'i':             indices,
        'seq':           epi_i['A_i'],
        'alpha':         alpha,
        'beta':          beta,
    })

    scores.loc[:, 'alpha_smooth'] = smooth(scores['alpha'], na)
    scores.loc[:, 'beta_smooth'] =  smooth(scores['beta'], nb)

    return(scores)


def alpha_score(epi_ij, i, std_i1, params=[1, 0.723, 0.635, 0.587], l=1, r=1):
    '''computes alpha score as {i+4} - {i+3} - {i+2} + {i+1},
    provided parameters for normalizing each {i+x} (e.g. the
    correlations identified for ECs in Toth-Petroczy et al.)'''
    i1 = (l*epi_ij[(i-1, i)] + r*epi_ij[(i, i+1)]) / (params[0])
    i2 = (l*epi_ij[(i-2, i)] + r*epi_ij[(i, i+2)]) / (params[1])
    i3 = (l*epi_ij[(i-3, i)] + r*epi_ij[(i, i+3)]) / (params[2])
    i4 = (l*epi_ij[(i-4, i)] + r*epi_ij[(i, i+4)]) / (params[3])
    return((i4 + i3 - i2 - i1)/std_i1)


def beta_score(epi_ij, i, std_i1, params=[1, 0.723], l=1, r=1):
    '''computes alpha score as {i+2} - {i+1},
    provided parameters for normalizing each {i+x}'''
    i1 = (l*epi_ij[(i-1, i)] + r*epi_ij[(i, i+1)]) / (params[0])
    i2 = (l*epi_ij[(i-2, i)] + r*epi_ij[(i, i+2)]) / (params[1])
    return((i2 - i1)/std_i1)


def smooth(vals, n=1, offset=None):
    '''average values over a range'''
    vals = list(vals)
    if offset is None:
        offset = int(np.around(n//2)) - 1

    sm = [np.nan]*offset
    for i in range(len(vals)-n):
        sm.append(np.mean(vals[i:i+n]))
    sm += [np.nan]*(n-offset)

    return(sm)


# secondary structure visualization ######################################


def plot_alpha_pairs(a, ss, indices, o=0, fs=12):
    '''plot alphascores, and measurement depth of corresponding pairs'''
    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(10, 8)

    # bottom subplot: number of mutants at position i, i+x
    i_3 = a[a.apply(lambda x: np.abs(x['i']-x['j'])==3,axis=1)]
    i_4 = a[a.apply(lambda x: np.abs(x['i']-x['j'])==4,axis=1)]

    ax[2].hist(i_3['i'], rwidth=0.25, bins=indices, color='green')
    ax[2].hist(i_4['i']+0.5, rwidth=0.25, bins=indices+0.25, color='gray')

    # middle subplot: largest epistasis measured at position i, i+x
    i_3 = i_3.sort_values('epi').drop_duplicates('i')
    i_4 = i_4.sort_values('epi').drop_duplicates('i')

    ax[1].scatter(i_3['i'],i_3['epi'], color='green')
    ax[1].scatter(i_4['i'],i_4['epi'], color='gray')

    # top subplot: secondary structure scores
    ax[0].scatter(indices, ss.set_index('i')['alpha'].loc[indices], color='black', s=15)
    ax[0].plot(indices, ss.set_index('i')['alpha_smooth'].loc[indices], color='green')
    ax[0].axhline(0, color='black')

    # formatting niceties:
    ax[2].set_ylim(0, 361)
    ax[2].legend(['i+1','i+2'], fontsize=12)
    ax[2].set_ylabel('observed mutants at pair (i,j)',fontsize=12)
    ax[2].set_xlabel('position (i)',fontsize=15)
    ax[1].set_ylabel('max epistasis value', fontsize=12)
    ax[0].set_ylabel('alpha score', fontsize=12)
    ax[0].legend(['smoothed score [i, i+1]', 'score at i'])

    for a in [0, 1, 2]:
        ax[a].set_xlim(indices[0], indices[-1])
        ax[a].set_xticks(indices+0.5)
        ax[a].grid(which='major', axis='x', linestyle='--')
        if a == 2:
            ax[2].set_xticklabels(indices+o, rotation=45, fontsize=fs)
        else:
            ax[a].set_xticklabels([], rotation=45, fontsize=12)
            ax[a].axhline(0,color='black')
    return(fig, ax)


def plot_beta_pairs(a, ss, indices, o=0, fs=12):
    '''plot betascores, and measurement depth of corresponding pairs'''
    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(10, 8)

    # bottom subplot: number of mutants at position i, i+x
    i_1 = a[a.apply(lambda x: np.abs(x['i']-x['j'])==1,axis=1)]
    i_2 = a[a.apply(lambda x: np.abs(x['i']-x['j'])==2,axis=1)]

    ax[2].hist(i_1['i'], rwidth=0.25, bins=indices, color='red')
    ax[2].hist(i_2['i']+0.5, rwidth=0.25, bins=indices+0.25, color='pink')

    # middle subplot: largest epistasis measured at position i, i+x
    i_1 = i_1.sort_values('epi', ascending=False).drop_duplicates('i')
    i_2 = i_2.sort_values('epi', ascending=False).drop_duplicates('i')

    ax[1].scatter(i_1['i'], i_1['epi'], color='red')
    ax[1].scatter(i_2['i'], i_2['epi'], color='pink')

    # top subplot: secondary structure scores
    ax[0].scatter(indices, ss.set_index('i')['beta'].loc[indices], color='black', s=15)
    ax[0].plot(indices, ss.set_index('i')['beta_smooth'].loc[indices], color='red')
    ax[0].axhline(0, color='black')

    # formatting niceties:
    ax[2].set_ylim(0, 361)
    ax[2].legend(['i+1','i+2'], fontsize=12)
    ax[2].set_ylabel('observed mutants at pair (i,j)',fontsize=12)
    ax[2].set_xlabel('position (i)',fontsize=15)
    ax[1].set_ylabel('max epistasis value', fontsize=12)
    ax[0].set_ylabel('beta score', fontsize=12)
    ax[0].legend(['smoothed score [i, i+1]', 'score at i'])

    for a in [0, 1, 2]:
        ax[a].set_xlim(indices[0], indices[-1])
        ax[a].set_xticks(indices+0.5)
        ax[a].grid(which='major', axis='x', linestyle='--')
        if a == 2:
            ax[2].set_xticklabels(indices+o, rotation=45, fontsize=fs)
        else:
            ax[a].set_xticklabels([], rotation=45, fontsize=12)
            ax[a].axhline(0, color='black')

    return(fig, ax)


def minimum_ss_unit(x, n=2):
    '''from a given secondary structure,
    drop ss elements less than a minimum length'''
    x = x + ['']
    ss = []
    cur = []
    for a in x:
        if (len(cur) == 0) or (a == cur[-1]):
            cur += [a]
        elif len(cur) > n:
            ss += cur
            cur = [a]
        else:
            ss += ['C']*len(cur)
            cur = [a]
    return(ss)


# Subsampling sparse mutant libraries #########################################


def draw_samples(data, M, N, sele_fxn):
    '''draw M rows from data table, N times, at random'''
    samples = []
    m = np.zeros(len(data), dtype=int)
    m[:M] = 1

    for n in range(N):
        r = deepcopy(m)
        np.random.shuffle(r)

        # draw samples from data, apply scoring function
        sample_n = data.loc[r.astype(bool)]
        sample_n = sele_fxn(sample_n)
        samples.append(sample_n)

    return(samples)


def sample_precisions(dataset, library_sizes, num_draws=10):
    '''randomly sample epistasis pairs from dataset,
    at various library sizes, for n independent draws,
    compute and store resulting 3D precision of most epistatic pairs'''
    sampling_results = []

    for n in library_sizes:
        samples = draw_samples(
            dataset, n, num_draws,
            lambda x: most_epistatic_pairs(x)
        )

        for i, sample in enumerate(samples):
            dists = sample['dist.any_struct']
            LR = sample['LR']
            prec = (
                n, np.mean(dists.head(28).lt(5)),
                np.mean(dists.head(56).lt(5)),
                np.mean(dists[LR].head(28).lt(5)),
                np.mean(dists[LR].head(56).lt(5)))

        sampling_results.append(prec)

    cols = ['library size', 'L/2', 'L', 'L/2 (i-j > 5)', 'L (i-j > 5)']
    sampling_results = pd.DataFrame(sampling_results, columns=cols)

    return(sampling_results)


def sample_constraints(dataset, library_sizes, num_draws=10):
    '''gets top epistasis pairs from random draws of various library sizes,
    for each library size, returns a list of tables with the top most epistatic i,j pairs'''
    sampling_pairs = {}

    for n in library_sizes:
        samples = draw_samples(
            dataset, n, num_draws,
            lambda x: pair_format(most_epistatic_pairs(x))
        )
        sampling_pairs[n] = samples

    return(sampling_pairs)


def pair_format(table, epi='epi'):
    '''return formatted table for folding'''
    pairs = {
        'i': table['positions'].apply(lambda x: x[0]),
        'j': table['positions'].apply(lambda x: x[1]),
        'A_i': table['mut'].apply(lambda x: x[0][0]),
        'A_j': table['mut'].apply(lambda x: x[1][0]),
        'cn': table[epi]
    }
    return(pd.DataFrame(pairs))


# blindly ranking folded models ##############################################

def sym_mat(triangle_file):
    '''load symmetric matrix from triangular table'''
    triangle_mat = pd.read_csv(triangle_file, index_col=0)
    square_mat = deepcopy(triangle_mat)
    for i, row in triangle_mat.iterrows():
        for k, v in row.items():
            if not pd.isnull(v):
                square_mat.loc[k, i] = v

    return(square_mat)


def weights_table(constraints, L, cb95, cb975):
    '''load in weights for scoring constraints, according to C-beta dictionary'''
    pairs = pd.read_csv(constraints).head(L)
    w = pairs['epi'] - np.min(pairs['epi'])

    pairs.loc[:, 'weight'] = w/(2*np.mean(w)) + 0.5
    pairs.loc[:, 'nu'] = pairs.apply(
        lambda x: cb95[x['A_i']][x['A_j']], axis=1)
    pairs.loc[:, 'ka'] = pairs.apply(
        lambda x: cb975[x['A_i']][x['A_j']], axis=1)
 
    return(pairs)


def load_backbone_constraints(strands, pairs):
    '''load in which strand-strand constraints to score'''
    s = pd.read_csv(strands)
    s.loc[:, 'nu'] = 6
    s.loc[:, 'ka'] = 2
    
    p = pd.read_csv(pairs)
    p = set(list(zip(p['i'], p['j'])) + list(zip(p['j'], p['i'])))

    s.loc[:, 'weight'] = s.apply(lambda x: tuple([x['i'], x['j']]) in p, axis=1).astype(int)
    
    return(s)


def sigmoid(d, nu, ka):
    '''sigmoid activator function'''
    if d < nu:
        f = 1
    else:
        f = np.exp(-((d - nu)**2) / (2*(ka**2)))
    return(f)


def pair_score(dists, pairs):
    '''computes weighted sigmoid score across pairs'''
    pairs.loc[:, 'dist'] = dists[:len(pairs)]
    s = pairs.apply(lambda x: sigmoid(x['dist'], x['nu'], x['ka']), axis=1)
    return(np.sum(pairs['weight'] * s))


def read_maxclust(filename):
    '''read maxcluster rmsd output into a table'''
    lines = pd.read_csv(filename, names=['line'], sep=';', skiprows=range(5))
    files = lines['line'].apply(lambda x: x.split()[5])
    rmsds = lines['line'].apply(
        lambda x: float(x.split('=')[1].strip().split(' ')[0]))

    table = pd.DataFrame({'filename': files, 'rmsd': rmsds})

    return(table)


def read_dists(table):
    '''parses distance lists nested in dataframe'''
    for col in table:
        if ('dist' in col):
            table.loc[:, col] = table[col].apply(lambda x: literal_eval(x))
    return(table)


def compare_tbl(RMSD, DIST, PAIRS, L, cb95, cb975, STRAND=None, SHEET=None):
    '''compute ranking scores from residue-residue distances, and merge with rmsd table'''
    rm = read_maxclust(RMSD)
    di = pd.read_csv(DIST, sep=';')
    di = read_dists(di)

    rm.loc[:, 'pdb_name'] = rm['filename'].apply(lambda x: x.split('/')[-1])
    di.loc[:, 'pdb_name'] = di['filename'].apply(lambda x: x.split('/')[-1])
    comp = pd.merge(di, rm, on='pdb_name')
    comp = constraint_scores(comp, PAIRS, L, cb95, cb975, STRAND, SHEET)    
    return(comp)


def constraint_scores(dists, PAIRS, L, cb95, cb975, STRAND, SHEET):
    '''computes secondary structure, 3D pair, and sheet scores'''
    if isinstance(dists, str):
        dists = pd.read_csv(dists, sep=';')
        dists = read_dists(dists)

    dists = dists.rename(columns={'ranking_score': 'ss_score'})
    pairs = weights_table(PAIRS, L, cb95, cb975)
    sheet = load_backbone_constraints(STRAND, SHEET)
    N = np.sum(sheet['weight'])

    dists.loc[:, 'pair_score'] = dists['cb_dists'].apply(
        lambda x: pair_score(x, pairs))/L
    dists.loc[:, 'sheet_score'] = dists['bb_dists'].apply(
        lambda x: pair_score(x, sheet))/N

    # each term in score is normalized
    # to be between 0 and 1 (maximum possible), and added in 3 equal parts
    dists.loc[:, 'score'] = \
        (dists['ss_score'] + dists['pair_score'] + dists['sheet_score'])/3

    return(dists)


# Plotting modeling and ranking results #################################

def set_axis(ax, xs, ys, xlims, ylims, xls=None, yls=None):
    '''format axes'''
    ax.set_xticks(xs)
    ax.set_yticks(ys)
    if xls is not None:
        ax.set_xticklabels(xls, fontsize=14)
    if yls is not None:
        ax.set_yticklabels(yls, fontsize=14)
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.invert_xaxis()


# Reporting precision #######################################################

def precision_table(pairs, Ls):
    '''compute precisions at various number of pairs (any, and long-range),
    also computes p-values according to the hypergeometric test'''
    dists = [k for k in pairs.keys() if 'dist' in k]
    LR = pairs[pairs['LR']]

    M = len(pairs)
    n = np.sum(pairs['dist.any_struct'].lt(5))
    LR_M = len(LR)
    LR_n = np.sum(pairs['dist.any_struct'].lt(5))

    prec = {}
    for L in Ls:
        prec[str(L)] = {}
        prec['LR '+str(L)] = {}

        for d in dists:
            h = np.sum(pairs[d].head(L).lt(5))
            LR_h = np.sum(LR[d].head(L).lt(5))

            prec[str(L)][d] = str(100 * h/L)[:4] + '%'
            prec['LR '+str(L)][d] = str(100 * LR_h/L)[:4] + '%'

        h = np.sum(pairs['dist.any_struct'].head(L).lt(5))
        LR_h = np.sum(LR['dist.any_struct'].head(L).lt(5))

        prec[str(L)]['p-value (any struct)'] = hypergeom.sf(h-1, M, n, L)
        prec['LR '+str(L)]['p-value (any struct)'] = hypergeom.sf(LR_h-1, LR_M, LR_n, L)

    prec = pd.DataFrame(prec)
    return(prec)


# Other epistasis -> structure methods, not used in results of paper ##########

def dedup_sites(tbl, n):
    '''limits number of pairs per position'''
    pos = set.union(set(tbl['positions'].apply(lambda x: x[0])),
                    set(tbl['positions'].apply(lambda x: x[1])))
    co = {i: 0 for i in pos}
    new_tbl = []
    for k, row in tbl.iterrows():
        i = row['positions'][0]
        j = row['positions'][1]

        if (co[i] < n) and (co[j] < n):
            new_tbl.append(row)
            co[i] += 1
            co[j] += 1

    return(pd.DataFrame(new_tbl))


def APC(pairs, e='epi'):
    '''compute average-product correction of pair-wise metric'''
    pos = set.union(set(pairs['positions'].apply(lambda x: x[0])),
                    set(pairs['positions'].apply(lambda x: x[1])))
    i_avg = {
        i: np.nanmean(
            pairs[pairs['positions'].apply(
                lambda x: i in set(x)
            )][e])
        for i in pos
    }
    ij_avg = np.nanmean(pairs['epi'])

    apc = deepcopy(pairs)
    apc.loc[:, e+'.apc'] = pairs.apply(
        lambda x: x[e] - i_avg[x['i']]*i_avg[x['j']]/ij_avg, axis=1)

    return(apc.sort_values(e+'.apc', ascending=False))
