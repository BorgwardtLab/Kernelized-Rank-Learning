#!/usr/bin/env python

import sys
import argparse
import matplotlib as mpl
import os
if os.path.isfile('.pyplot_noX'): mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sb
import numpy as np
from plot_utils import PlotsError
import plot_utils

def get_bins(data_group, nbins=None, binw=None, minimum=None, maximum=None):
	if nbins is None and binw is None: return None, None
	if nbins is not None and binw is not None: raise ValueError
	
	#if all lists are empty...
	if sum(map(len, data_group)) == 0:
		a, b = 0, 0
	else:
		a = min(map(min, data_group))
		b = max(map(max, data_group))
	
	if minimum is not None: a = minimum
	if maximum is not None: b = maximum
	
	if nbins is not None:
		if a == b: a -= 0.5; b += 0.5
		bins = np.linspace(a, b, nbins+1, endpoint=True)
	else:
		if a == b: a -= binw / 2.0; b += binw / 2.0
		bins = np.arange(a, b + binw, binw)
	
	if len(bins) < 2: raise ValueError
	if binw is not None and round(binw, 4) != round(bins[1] - bins[0], 4): raise ValueError
	return bins, bins[1] - bins[0]

def factorise(data, factor_index=0):
	factorised = {}
	for d in data:
		if d[factor_index] not in factorised: factorised[d[factor_index]] = []
		factorised[d[factor_index]].append(d)
	if len(data) != sum([len(factorised[f]) for f in factorised]): raise ValueError
	return factorised

#if factors are specified, all groups must be of 1 only
def parse_data(data_descriptors, groups, factors=None, auto_labels=False, sep='\t'):
	#data_descriptors and factors can be in two different formats:
	#1) fn1?colA?colB?colC -> read all the columns of the file at once
	#2) fn1?colA fn2?colB ... -> read the files column by column as you go
	if len(data_descriptors) == 1 and data_descriptors[0].count('?') > 1:
		data_pre_parsed = True
		data_descriptors_split = data_descriptors[0].split('?')
		df_data = plot_utils.read_table(data_descriptors_split[0], usecols=map(int, data_descriptors_split[1:]), sep=sep)
		data_descriptors = [data_descriptors_split[0] + '?' + column for column in data_descriptors_split[1:]]
	else:
		data_pre_parsed = False
		
	if factors is not None:
		if len(factors) == 1:
			#the special case of factors[0].count('?') == 1 needs to be pre-processed as well
			#because factors allow to specify a single factors column for multiple data_items!
			factors_pre_parsed = True
			factors_split = factors[0].split('?')
			df_factors = plot_utils.read_table(factors_split[0], usecols=map(int, factors_split[1:]), sep=sep)
			factors = [factors_split[0] + '?' + column for column in factors_split[1:]]
		else:
			factors_pre_parsed = False
	
	grouped_data_descriptors, i = [], 0
	for g in groups:
		grouped_data_descriptors.append(data_descriptors[i:i + g])
		i += g
	if i != len(data_descriptors): raise ValueError
	
	data, labels = [], []
	for i, data_descriptors_group in enumerate(grouped_data_descriptors):
		if factors is not None and len(data_descriptors_group) != 1: raise ValueError
		data_group = []
		for data_descriptor in data_descriptors_group:
			data_filename, data_column = data_descriptor.split('?')
			data_column = int(data_column)
			if not data_pre_parsed: df_data = plot_utils.read_table(data_filename, usecols=[data_column], sep=sep)
			if factors is None:
				data_group.append(df_data[data_column].tolist())
				labels.append(data_descriptor if not auto_labels else read_labels([data_descriptor], sep=sep)[0])
			else:
				factors_filename, factors_column = factors[i].split('?') if len(factors) != 1 else factors[0].split('?')
				factors_column = int(factors_column)
				if not factors_pre_parsed: df_factors = plot_utils.read_table(factors_filename, usecols=[factors_column], sep=sep)
				if len(df_factors[factors_column]) != len(df_data[data_column]): raise ValueError
				factorised = factorise(zip(df_factors[factors_column].tolist(), df_data[data_column].tolist()))
				for factor in sorted(factorised.keys(), reverse=True):
					data_group.append(map(lambda x: x[1], factorised[factor]))
					labels.append((data_descriptor if not auto_labels else read_labels([data_descriptor], sep=sep)[0]) + ' (' + str(factor) + ')')
		data.append(data_group)
	
	return data, labels

def read_labels(data_descriptors, sep='\t'):
	labels = []
	first_lines = {}
	for d in data_descriptors:
		data_filename, data_column = d.split('?')
		if data_filename not in first_lines:
			with open(data_filename, 'r') as f: first_line = f.readline()
			if first_line[0] != '#': raise ValueError
			first_lines[data_filename] = first_line
		label = first_lines[data_filename][1:].split(sep)[int(data_column)]
		labels.append(label[:label.rfind(':')])
	return labels

def plot_hist(data, data_descriptors=None, labels=None, auto_labels=False, y_tics=None, linewidth=2, nbins=None, binw=None, groups=None, factors=None, ncols=None,\
			x_range=None, y_range=None, x_log=False, y_log=False, sep='\t', hist=True, kde=True, rug=True, normed=True, title=None, x_label=None, y_label=None, rotate_x_tics=None, bold_x_tics=False,\
			hide_x_tick_marks=False, hide_y_tick_marks=False,\
			hide_x_ticks=False, hide_y_ticks=False,\
			label=None, show_binw=True, show_legend=True, legend_out=False, legend_out_pad=None, despine=True, style='despine_ticks',\
			fontsize=16, colours=None, palette=None, reverse_palette=False, ncolours=None, figsize=None, fig_padding=0.1, dpi=None, output=None, out_format=None, legend_kwargs=None, kwargs=None):
	'''
	Parameters
	----------
	data : list of lists (subplots) of lists (histograms)
	       [ [ [1, 1, 2, 2] ], [ [0.1, 0.2, 0.3], [0.3, 0.4, 0.5] ] ]
	       --------------------------data----------------------------
	         ---subplot_1----  --------------subplot_2-------------
	           -histogram-       ---histogram---  ---histogram--- 
	'''
	if data is not None:
		if data_descriptors is not None: raise PlotsError(message='You can specify only one of the mutually exclusive arguments: "data" or "data_descriptors"')
		if groups is not None: raise PlotsError(message='You cannot specify "groups" if "data" is specified"')
		if factors is not None: raise PlotsError(message='You cannot specify "factors" if "data" is specified')
		data_items_count = sum([len(x) for x in data])
	else:
		if data_descriptors is None: raise PlotsError(message='You must specify "data" or "data_descriptors"')
		if len(data_descriptors) == 1:
			data_items_count = data_descriptors[0].count('?')
		else:
			data_items_count = len(data_descriptors)
			for d in data_descriptors:
				if d.count('?') != 1: raise PlotsError(message='data filenames and 0-based columns must be in this format: filenameA?columnX')
	
	if groups is not None and factors is not None: raise PlotsError(message='You can specify only one of the mutually exclusive arguments: "groups" or "factors"')
	if groups is not None and sum(groups) != data_items_count: raise PlotsError(message='Number of data items specified as "data_descriptors" must be equal to the sum of counts specified by "groups"')	
	if auto_labels and labels is not None: raise PlotsError(message='You cannot specify "labels" if "auto_labels" is set')
	
	if factors is not None: 
		if len(factors) == 1:
			factors_count = factors[0].count('?')
		else:
			factors_count = len(factors)
			for f in factors:
				if f.count('?') != 1: raise PlotsError(message='factor filenames and 0-based columns must be in this format: filenameA?columnX')
		if factors_count != 1 and factors_count != data_items_count: raise PlotsError(message='Number of data items specified as "data_descriptors" must be equal to the number of factors specified as "factors"')
	
	if labels is not None and factors is None and len(labels) != data_items_count: raise PlotsError(message='Number of data items specified as "data_descriptors" must be equal to the number of labels specified as "label"')
	
	if not show_legend and (legend_out or legend_kwargs != None): raise PlotsError(message='If you hide the legend (--hide_legend or show_legend=False), you cannot set it outside (legend_out) or set it properties (legend_kwargs)')
	for x_y_range in [(x_range, 'xrange'), (y_range, 'yrange')]:
		if x_y_range[0] is not None and (len(x_y_range[0]) != 2 or x_y_range[0][0] >= x_y_range[0][1]): raise PlotsError(message='You need to provide exactly two numbers to set %s: "min max"' % x_y_range[1])
	if nbins is not None and binw is not None: raise PlotsError(message='You can specify only one of the mutually exclusive arguments: "bins" or "binw"')
	if colours is not None:
		if palette is not None: raise PlotsError(message='You can specify only one of the mutually exclusive arguments: "colours" or "palette"')
		if ncolours is not None: raise PlotsError(message='You cannot specify "ncolours" when you specified "colours"')
	if figsize is not None and len(figsize) != 2: raise PlotsError(message='You need to provide exactly two numbers to set figure size: "width height"')
	if not hist and not kde and not rug: raise PlotsError(message='You need to plot at least one of a) histogram or b) KDE density or c) a rug plot')
		
	#if nbins is None and binw is None: print 'INFO: calculating the number of histogram bins using the Freedman-Diaconis rule'
	if y_label is None: y_label = 'Density' if kde or normed else 'Frequency'
	if ncols is None:
		if groups is not None: 
			ncols = len(groups) if len(groups) < 4 else 2
		elif factors is not None:
			ncols = data_items_count if data_items_count < 4 else 2
		else:
			ncols = 1
	
	if data is None:
		if groups is not None:
			force_groups = groups
		elif factors is not None:
			force_groups = [1] * data_items_count
		else:
			force_groups = [data_items_count]
		data, parsed_labels = parse_data(data_descriptors, groups=force_groups, factors=factors, auto_labels=auto_labels, sep=sep)
		if labels is None: labels = parsed_labels
		if len(labels) != len(parsed_labels): raise ValueError

	font = plot_utils.init_plot_style(style, fontsize, colours, palette, reverse_palette, ncolours, hide_x_tick_marks=hide_x_tick_marks, hide_y_tick_marks=hide_y_tick_marks)
	nrows = int(len(data) / ncols) + (len(data) % ncols != 0)
	fig, axs = plt.subplots(nrows, ncols, sharex=False, sharey=False, squeeze=False)
	if figsize is not None: fig.set_figwidth(figsize[0]); fig.set_figheight(figsize[1])
	if dpi is not None: fig.set_dpi(dpi)
	
	artists, l = [], 0
	if label is None: label = ''
	original_label = label
	for i, group_data in enumerate(data):
		label = original_label
		r = int(i / ncols)
		c = i % ncols
		bins, binw = get_bins(group_data, nbins=nbins, binw=binw if nbins is None else None, minimum=x_range[0] if x_range is not None else None, maximum=x_range[1] if x_range is not None else None)
		last_l = l
		for data_list in group_data:
			default_kwargs = dict(a=data_list, ax=axs[r][c], hist=hist, kde=kde, rug=rug, norm_hist=normed, label=labels[l])
			try:
				sb.distplot(kde_kws={'lw': linewidth}, hist_kws={'linewidth': linewidth}, **plot_utils.merged_kwargs(default_kwargs, dict(bins=bins), kwargs))
				l += 1
			except ValueError:
				l = last_l
				label += '\nWARNING: number of bins is 10'
				for data_list in group_data:
					sb.distplot(kde_kws={'lw': linewidth}, hist_kws={'linewidth': linewidth}, **plot_utils.merged_kwargs(default_kwargs, dict(bins=10), kwargs))
					l += 1
				break
			
		if x_range is not None: axs[r][c].set_xlim(x_range[0], x_range[1])
		if y_range is not None: axs[r][c].set_ylim(y_range[0], y_range[1])
		if y_tics is not None: axs.set_yticks(y_tics)
		if x_log: axs[r][c].set_xscale('log')
		if y_log: axs[r][c].set_yscale('log')
		if x_label is not None: axs[r][c].set_xlabel(x_label, labelpad=15, fontproperties=font.get('b'))
		if y_label is not None: axs[r][c].set_ylabel(y_label, labelpad=15, fontproperties=font.get('b'))
		if title is not None: axs[r][c].set_title(title, fontproperties=font.get('b'))
		#if show_binw and binw is not None: axs[r][c].text(0.05, 0.95 if label is None else 0.9, 'bin width ' + str(binw if nbins is None else round(binw, 4)), fontsize=fontsize, horizontalalignment='left', verticalalignment='top', transform=axs[r][c].transAxes)
		if show_binw and binw is not None: label += ('\n\n' if label is not None else '') + 'bin width ' + str(binw if nbins is None else round(binw, 4))
		if label is not None: axs[r][c].text(0.05, 0.95, label, horizontalalignment='left', verticalalignment='top', transform=axs[r][c].transAxes, fontproperties=font.get('n'))
		if show_legend: axs[r][c].legend(prop=font.get('n'))
		#Legend is a little tricky if the option is to be outside of the plot
		if show_legend:
			legend_out_kwargs = dict(bbox_to_anchor=(1, 1), loc=2, borderpad=legend_out_pad) if legend_out else None
			legend = axs[r][c].legend(**plot_utils.merged_kwargs(dict(prop=font.get('n')), legend_out_kwargs, legend_kwargs))
			artists.append(legend)
		plt.setp(axs[r][c].get_xticklabels(), rotation=rotate_x_tics)
		plot_utils.set_fontproperties(font.get('b' if bold_x_tics else 'n'), axs[r][c].get_xticklabels())
		plot_utils.set_fontproperties(font.get('n'), axs[r][c].get_yticklabels())
		if hide_x_ticks: axs[r][c].xaxis.set_major_locator(ticker.NullLocator())
		if hide_y_ticks: axs[r][c].yaxis.set_major_locator(ticker.NullLocator())

	for empty in range(c + 1, ncols):
		axs[r][empty].axis('off')

	if despine or style in ['despine_ticks', 'whitegrid_ticks']: sb.despine(top=True, right=True)
	
	plt.tight_layout()
	if output is not None:
		plt.savefig(output, format=out_format, dpi=dpi, additional_artists=artists, bbox_inches='tight', pad_inches=fig_padding)
	else:
		plt.show()
	plt.close()
	
def suggest_nbins(a, max_nbins=20):
	return min(sb.distributions._freedman_diaconis_bins(a), max_nbins)

def main():
	print '#', ' '.join(sys.argv)
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('data_descriptors', nargs='+', type=str, help='list of data filenames and 0-based columns in this format: 1) multiple files: filenameA?columnX filenameB?columnY ... or 2) single file: filenameA?columnX?columnY?columnZ (please note different behaviour depending on if --groups or --factors are specified)')
	parser.add_argument('--groups', nargs='+', type=int, help='list of numbers to represent how many data items from "data_descriptors" to plot into a single plot (mutually exclusive with --factors); DEFAULT: "None" means all data items specified in "data_descriptors" into a single plot; USAGE: "fnA?colX fnB?colY fnC?colZ --groups 2 1" will produce two plots with the first plot plotting two data items "fnA?colX fnB?colY" and the second plot plotting 1 data item "fnC?colZ"')
	parser.add_argument('--factors', nargs='+', type=str, help='list of factors filenames and 0-based columns in this format: 1) multiple files: filenameF?columnK filenameG?columnL ... or 2) single file: filenameF?columnK?columnL?columnM or 3) single file with a single factor column: filenameF?columnK; this means that each data item specified in "data_descriptors" will produce its own plot with several histograms based on the defined factors (mutually exclusive with --groups); DEFAULT: "None" means there are no factors and each data item produces a single histogram; USAGE: "fnA?colX fnB?colY fnC?colZ --factors fnF?colK fnG?colL fnH?colM" will make three plots each with as many histogram as there are factors in fnF?colK, fnG?colL, and fnH?colM')
	parser.add_argument('--labels', nargs='+', type=str, help='labels for the data: label1 label2 ...; the number of labels needs to be equal to the number of data items in "data_descriptors" but if --factors are specified, each factor needs its own label for each data item')
	parser.add_argument('--auto_labels', action='store_true', help='labels for the data will be parsed from the first line of the file which has to start with "#"')
	parser.add_argument('--nbins', type=int, help='how many bins to plot in the histogram (mutually exclusive with --binw)')
	parser.add_argument('--binw', type=float, help='width of the bin (mutually exclusive with --nbins)')
	parser.add_argument('--no_hist', action='store_true', help='do NOT plot histogram')
	parser.add_argument('--kde', action='store_true', help='to add a density plot (estimated with KDE)')
	parser.add_argument('--rug', action='store_true', help='to add a rug plot')
	parser.add_argument('--normed', action='store_true', help='to plot density of a histogram instead of frequency (if --kde is set, density is always plotted)')
	parser.add_argument('--hide_binw', action='store_true', help='hide the bin width text label')
	parser.add_argument('--ncols', type=int, help='how many plots should be in a single row')
	plot_utils.add_common_arguments(parser, 'despine_ticks')
	args = parser.parse_args()
	try:
		plot_hist(data=None, data_descriptors=args.data_descriptors, labels=args.labels, auto_labels=args.auto_labels, y_tics=args.ytics, linewidth=args.linewidth, nbins=args.nbins, binw=args.binw, groups=args.groups, factors=args.factors, ncols=args.ncols, x_range=args.xrange, y_range=args.yrange, x_log=args.xlog, y_log=args.ylog, sep=args.sep, hist=not args.no_hist, kde=args.kde, rug=args.rug, normed=args.normed, title=args.title, x_label=args.xlabel, y_label=args.ylabel, rotate_x_tics=args.rotate_xtics, bold_x_tics=args.bold_xtics, hide_x_tick_marks=args.hide_xtick_marks, hide_y_tick_marks=args.hide_ytick_marks, hide_x_ticks=args.hide_xticks, hide_y_ticks=args.hide_yticks, label=args.label, show_binw=not args.hide_binw, show_legend=not args.hide_legend, legend_out=args.legend_out, legend_out_pad=args.legend_out_pad, despine=args.despine, style=args.style, fontsize=args.fontsize, colours=args.colours, palette=args.palette, reverse_palette=args.reverse_palette, ncolours=args.ncolours, figsize=args.figsize, fig_padding=args.fig_padding, dpi=args.dpi, output=args.output, out_format=args.format, kwargs=plot_utils.str_to_kwargs(args.kwargs) if args.kwargs is not None else None)
	except PlotsError as e:
		parser.error(e.message)
	
if __name__ == '__main__':
	main()
