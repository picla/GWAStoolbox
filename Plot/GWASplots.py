#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import pylab 
import scipy as sp
from scipy import stats

'''
List of functions for making often used plots in python
'''

__version__ = "0.0.1"
__updated__ = "09.08.2017"
__date__ = "09.08.2017"



SUPPORTED_FORMAT=('png','pdf')

# Absolute position
def _absolutePosition(chromosome, position):
	chromosome = int(chromosome)
	chrLength = [0, 30427700, 19698300, 23460000, 18585000, 26975600]
	spacer = 10**6
	absPos = position + np.cumsum(chrLength[0:chromosome])[-1] + (chromosome - 1) * spacer
	return (int(absPos))

# Mahattan Plot
# requires prefiltered data (no mac or maf filtering included)
# format: chromsome, position, p-value
#TODO: make plot broader
def manhattanPlot(GWASdata, outputFile):
	GWASdata.loc[:, 'absolutePosition'] = GWASdata.apply(lambda x: _absolutePosition(x['chromosome'], x['position']), axis = 1)
	GWASdata.loc[:, 'logPval'] = GWASdata.apply(lambda x: math.log10(x['Pval']) * -1, axis = 1)
	
	# get the halfway point of each chromosome in total position length (for plotting the xtick labels
	spacer = 10**6
	chrLength = [0, 30427700, 19698300, 23460000, 18585000, 26975600]
	absChrmHalf = [(sum(chrLength[0:chrm]) + (spacer * (chrm - 1)) + chrLength[chrm]/2) for chrm in range(1,6)]	

	fig = plt.figure(figsize=(16,6))
	# create subplotting object (allow for the chromsomes to be plotted one by one as suboplots on the same position)
	ax = fig.add_subplot(1,1,1)
	ax.scatter(GWASdata.absolutePosition[GWASdata.chromosome == 1], GWASdata.logPval[GWASdata.chromosome == 1], color = 'blue', s = 4)
	ax.scatter(GWASdata.absolutePosition[GWASdata.chromosome == 2], GWASdata.logPval[GWASdata.chromosome == 2], color = 'dodgerblue', s = 4)
	ax.scatter(GWASdata.absolutePosition[GWASdata.chromosome == 3], GWASdata.logPval[GWASdata.chromosome == 3], color = 'blue', s = 4)
	ax.scatter(GWASdata.absolutePosition[GWASdata.chromosome == 4], GWASdata.logPval[GWASdata.chromosome == 4], color = 'dodgerblue', s = 4)
	ax.scatter(GWASdata.absolutePosition[GWASdata.chromosome == 5], GWASdata.logPval[GWASdata.chromosome == 5], color = 'blue', s = 4 )

	#set y and x limits, before drawing threshold lines.
	plt.ylim(ymin = 0)
	plt.xlim(xmin = -2 * spacer, xmax = sum(chrLength) + 6 * spacer)

	# Bonferroni threshold
	n = len(GWASdata.Pval)
	bonf = 0.05/n
	bonf = math.log10(bonf) * -1
	plt.axhline(y = bonf, ls = 'dashed', color = 'red')

	# labels and tick marks
	ax.set_ylabel('-log10(P-value)')

	ax.set_xticks(absChrmHalf)
	ax.set_xticklabels(['chrm 1', 'chrm 2', 'chrm 3', 'chrm 4', 'chrm 5'])
	
	# set tick marks outside
	ax.tick_params(direction='out')
	# only tick marks on x and y axes
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	# save the plot
	plt.savefig(outputFile)
	plt.close()



# QQ-plot
def get_quantiles(scores, num_dots=1000):
    """
    Uses scipy
    """
    scores = sp.copy(sp.array(scores))
    scores.sort()
    indices = [int(len(scores) * i / (num_dots + 2)) for i in range(1, num_dots + 1)]
    return scores[indices]

def _getExpectedPvalueQuantiles_(numQuantiles):
    quantiles = []
    for i in range(numQuantiles):
        quantiles.append(float(i) + 0.5 / (numQuantiles + 1))
    return quantiles

def get_log_quantiles(scores, num_dots=1000, max_val=5):
    """
    Uses scipy
    """
    scores = sp.copy(sp.array(scores))
    scores.sort()
    indices = sp.array(10 ** ((-sp.arange(1, num_dots + 1, dtype='single') / (num_dots + 1)) * max_val) \
                * len(scores), dtype='int')
    return -sp.log10(scores[indices])

def calculate_qqplot_data(pvals,num_dots=1000):
    max_val = -math.log10(min(pvals))
    quantiles = get_quantiles(pvals, num_dots=num_dots)
    exp_quantiles = _getExpectedPvalueQuantiles_(num_dots)
    log_quantiles = get_log_quantiles(pvals, num_dots=num_dots, max_val=max_val)
    exp_log_quantiles = sp.arange(1, num_dots + 1, dtype='single') / (num_dots + 1) * max_val

    quantiles_dict = {'quantiles':quantiles, 'exp_quantiles':exp_quantiles,
            'log_quantiles':log_quantiles, 'exp_log_quantiles':exp_log_quantiles}
    return quantiles_dict

def simple_log_qqplot(quantiles_list, png_file=None, pdf_file=None, quantile_labels=None, line_colors=None,
            max_val=5, title=None, text=None, plot_label=None, ax=None, **kwargs):
    storeFig = False
    if ax is None:
        f = plt.figure(figsize=(5.4, 5))
        ax = f.add_axes([0.1, 0.09, 0.88, 0.86])
        storeFig = True
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2.0)
    num_dots = len(quantiles_list[0])
    exp_quantiles = sp.arange(1, num_dots + 1, dtype='single') / (num_dots + 1) * max_val
    for i, quantiles in enumerate(quantiles_list):
        if line_colors:
            c = line_colors[i]
        else:
            c = 'b'
        if quantile_labels:
            ax.plot(exp_quantiles, quantiles, label=quantile_labels[i], c=c, alpha=0.5, linewidth=2.2)
        else:
            ax.plot(exp_quantiles, quantiles, c=c, alpha=0.5, linewidth=2.2)
    ax.set_ylabel("Observed $-log_{10}(p$-value$)$")
    ax.set_xlabel("Expected $-log_{10}(p$-value$)$")
    if title:
        ax.title(title)
    max_x = max_val
    max_y = max(map(max, quantiles_list))
    ax.axis([-0.025 * max_x, 1.025 * max_x, -0.025 * max_y, 1.025 * max_y])
    if quantile_labels:
        fontProp = matplotlib.font_manager.FontProperties(size=10)
        ax.legend(loc=2, numpoints=2, handlelength=0.05, markerscale=1, prop=fontProp, borderaxespad=0.018)
    y_min, y_max = plt.ylim()
    if text:
        f.text(0.05 * max_val, y_max * 0.9, text)
    if plot_label:
        f.text(-0.138 * max_val, y_max * 1.01, plot_label, fontsize=14)
    if storeFig == False:
        return
    if png_file != None:
        f.savefig(png_file)
    if pdf_file != None:
        f.savefig(pdf_file, format='pdf')


def simple_qqplot(quantiles_list, png_file=None, pdf_file=None, quantile_labels=None, line_colors=None,
            title=None, text=None, ax=None, plot_label=None, **kwargs):
    storeFig = False
    if ax is None:
        f = plt.figure(figsize=(5.4, 5))
        ax = f.add_axes([0.11, 0.09, 0.87, 0.86])
        storeFig = True
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2.0)
    num_dots = len(quantiles_list[0])
    exp_quantiles = sp.arange(1, num_dots + 1, dtype='single') / (num_dots + 1)
    for i, quantiles in enumerate(quantiles_list):
        if line_colors:
            c = line_colors[i]
        else:
            c = 'b'
        if quantile_labels:
            ax.plot(exp_quantiles, quantiles, label=quantile_labels[i], c=c, alpha=0.5, linewidth=2.2)
        else:
            ax.plot(exp_quantiles, quantiles, c=c, alpha=0.5, linewidth=2.2)
    ax.set_ylabel("Observed $p$-value")
    ax.set_xlabel("Expected $p$-value")
    if title:
        ax.title(title)
    ax.axis([-0.025, 1.025, -0.025, 1.025])
    if quantile_labels:
        fontProp = matplotlib.font_manager.FontProperties(size=10)
        ax.legend(loc=2, numpoints=2, handlelength=0.05, markerscale=1, prop=fontProp, borderaxespad=0.018)
    if text:
        f.text(0.2, 0.9, text, horizontalalignment='left')
    if plot_label:
        f.text(-0.151, 1.04, plot_label, fontsize=14)
    if storeFig == False:
        return
    if png_file != None:
        f.savefig(png_file)
    if pdf_file != None:
        f.savefig(pdf_file, format='pdf')

def plot_simple_qqplots_pvals(file_prefix, pvals_list, result_labels=None, line_colors=None,
            num_dots=1000, title=None, max_neg_log_val=5, text = None):
    """
    Plots both log QQ-plots and normal QQ plots.
    """
    qs = [get_quantiles(pvals_list)]
    log_qs = [get_log_quantiles(pvals_list)]
 #   for pvals in pvals_list:
 #       qs.append(get_quantiles(pvals, num_dots))
 #       log_qs.append(get_log_quantiles(pvals, num_dots, max_neg_log_val))
    simple_qqplot(qs, pdf_file = file_prefix + '_qq.pdf', quantile_labels=result_labels,
                line_colors=line_colors, num_dots=num_dots, title=title, text = text)
    simple_log_qqplot(log_qs, pdf_file = file_prefix + '_log_qq.pdf', quantile_labels=result_labels,
                line_colors=line_colors, num_dots=num_dots, title=title, max_val=max_neg_log_val, text = text)


