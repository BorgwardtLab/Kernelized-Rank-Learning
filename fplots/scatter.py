#!/usr/bin/env python

import sys
import argparse
import matplotlib as mpl
import os
from fcommons import ml_utils
if os.path.isfile('.pyplot_noX'): mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sb
import numpy as np
from plot_utils import PlotsError
import plot_utils

def parse_data(data_descriptors, auto_labels=False, sep='\t'):
	x, y, data_filename_x, data_filename_y = [], [], None, None
	if len(data_descriptors) == 1:
		data_filename_x, x_col, y_col = data_descriptors[0].split('?')
		data_filename_y = data_filename_x
	elif len(data_descriptors) == 2: 
		data_filename_x, x_col = data_descriptors[0].split('?')
		data_filename_y, y_col = data_descriptors[1].split('?')
	else:
		raise PlotsError(message='unexpected format: ' + str(data_descriptors))
	x_col, y_col = int(x_col), int(y_col)
		
	if data_filename_x == data_filename_y:
		df = plot_utils.read_table(data_filename_x, usecols=[x_col, y_col], sep=sep)
		x = df[x_col].tolist()
		y = df[y_col].tolist()
	else:
		x = plot_utils.read_table(data_filename_x, usecols=[x_col], sep=sep)[x_col].tolist()
		y = plot_utils.read_table(data_filename_y, usecols=[y_col], sep=sep)[y_col].tolist()

	x_label = (data_filename_x[data_filename_x.rfind(sep) + 1:] + '?' + str(x_col)) if not auto_labels else read_labels(data_filename_x, x_col, sep=sep)
	y_label = (data_filename_y[data_filename_y.rfind(sep) + 1:] + '?' + str(y_col)) if not auto_labels else read_labels(data_filename_y, y_col, sep=sep)
	
	''''x = np.array(x)
	x = -1 * np.log(x)
	x[x==np.inf] = 9.0
	x[x==-0.0] = 0.0
	x = list(x)'''
	
	''''x = np.array(x)
	disease4 = x==4.0
	disease5 = x==5.0
	x = -1 * np.log(x)
	x[x==np.inf] = 9.0
	x[x==-0.0] = 0.0
	x[disease4] = 10.0
	x[disease5] = 10.0
	x = list(x)'''
	
	'''y = -1 * np.log(np.array(y))
	y[y==np.inf] = 9.0
	y[y==-0.0] = 0.0
	y = list(y)'''
	
	return x, y, x_label, y_label

def read_labels(data_filename, column, sep='\t'):
	with open(data_filename, 'r') as f:
		first_line = f.readline()
	if first_line[0] != '#': raise ValueError
	return first_line.strip('\n').split(sep)[column]

def plot_scatter(x, y, data_descriptors=None, linewidth=2, y_tics=None, x_range=None, y_range=None, x_log=False, y_log=False, sep='\t', title=None, x_label=None, y_label=None, rotate_x_tics=None, bold_x_tics=False,\
				hide_x_tick_marks=False, hide_y_tick_marks=False,\
				hide_x_ticks=False, hide_y_ticks=False,\
				auto_labels=False, label=None, show_corr=True, show_legend=False, legend_out=False, legend_out_pad=None, fit_reg=False, ci=None, despine=True, style='despine_ticks',\
				fontsize=16, colours=None, palette=None, reverse_palette=False, ncolours=None, figsize=None, fig_padding=0.1, dpi=None, output=None, out_format=None, legend_kwargs=None, kwargs=None):
	
	if (x is None and y is not None) or (x is not None and y is None):
		raise PlotsError(message='one sample is None: ' + ('x is None' if x is None else 'y is None'))
	elif x is not None and y is not None:
		if len(x) != len(y):
			raise PlotsError(message='Non-matching number of plots')
		for xx, yy in zip(x, y):
			if len(xx) != len(yy):
				raise PlotsError(message='unpaired samples detected: x size: ' + str(len(xx)) + ' y size: ' + str(len(yy)))
		if data_descriptors is not None: raise PlotsError(message='You can specify only one of the mutually exclusive arguments: "x" and "y" or "data_descriptors"')
	else:
		if data_descriptors is None: raise PlotsError(message='You must specify "data" or "data_descriptors"')
		if len(data_descriptors) == 1:
			if data_descriptors[0].count('?') != 2: raise PlotsError(message='When using a single data_descriptor, you need to specify exactly two columns with ?')
		elif len(data_descriptors) == 2: 
			if data_descriptors[0].count('?') != 1 or data_descriptors[1].count('?') != 1: raise PlotsError(message='When using two data_descriptors, you need to specify exactly one column for each with ?')
		else:
			raise PlotsError(message='Either specify one data_descriptor with two columns (2 x ?) or two data_descriptors with one column each (1 x ?)')
	if auto_labels and (x_label is not None or y_label is not None): raise PlotsError(message='You cannot specify "xlabel" or "ylabel" if "auto_labels" is set')
	if ci is not None and (ci < 0 or ci > 100): raise PlotsError(message='"ci" must be None or within 0 and 100')
	
	if not show_legend and (legend_out or legend_kwargs != None): raise PlotsError(message='If you hide the legend (--hide_legend or show_legend=False), you cannot set it outside (legend_out) or set it properties (legend_kwargs)')
	for x_y_range in [(x_range, 'xrange'), (y_range, 'yrange')]:
		if x_y_range[0] is not None and (len(x_y_range[0]) != 2 or x_y_range[0][0] >= x_y_range[0][1]): raise PlotsError(message='You need to provide exactly two numbers to set %s: "min max"' % x_y_range[1])
	if colours is not None:
		if palette is not None: raise PlotsError(message='You can specify only one of the mutually exclusive arguments: "colours" or "palette"')
	if figsize is not None and len(figsize) != 2: raise PlotsError(message='You need to provide exactly two numbers to set figure size: "width height"')
	
	if x is None and y is None:
		x, y, parsed_label_x, parsed_label_y = parse_data(data_descriptors, auto_labels=auto_labels, sep=sep)
		if x_label is None: x_label = parsed_label_x
		if y_label is None: y_label = parsed_label_y
		x, y = [x], [y]
	
	font = plot_utils.init_plot_style(style, fontsize, colours, palette, reverse_palette, ncolours, hide_x_tick_marks=hide_x_tick_marks, hide_y_tick_marks=hide_y_tick_marks)
	fig = plt.figure()
	if figsize is not None: fig.set_figwidth(figsize[0]); fig.set_figheight(figsize[1])
	if dpi is not None: fig.set_dpi(dpi)
	for xx, yy in zip(x, y):
		default_kwargs = dict(x=np.array(xx), y=np.array(yy), data=None, x_estimator=None, x_bins=None, x_ci='ci', scatter=True, fit_reg=fit_reg, ci=ci, n_boot=1000, units=None, order=1, logistic=False, lowess=False, robust=False, logx=False, x_partial=None, y_partial=None, truncate=False, dropna=False, x_jitter=None, y_jitter=None, label=None, color=None, marker='o', ax=None)
		axs = sb.regplot(line_kws={'linewidth': linewidth}, **plot_utils.merged_kwargs(default_kwargs, kwargs))
	if x_range is not None: axs.set_xlim(x_range[0], x_range[1])
	if y_range is not None: axs.set_ylim(y_range[0], y_range[1])
	if y_tics is not None: axs.set_yticks(y_tics)
	if x_log: axs.set_xscale('log')
	if y_log: axs.set_yscale('log')
	if x_label is not None: axs.set_xlabel(x_label, labelpad=15, fontproperties=font.get('b'))
	if y_label is not None: axs.set_ylabel(y_label, labelpad=15, fontproperties=font.get('b'))
	if title is not None: axs.set_title(title, fontproperties=font.get('b'))
	if label is not None: axs.text(0.05, 0.95, label, horizontalalignment='left', verticalalignment='top', transform=axs.transAxes, fontproperties=font.get('n'))
	if show_corr and len(x) == 1:
		r, _ = ml_utils.calc_r(x[0], y[0])
		axs.text(0.85, 0.95, 'r', horizontalalignment='left', verticalalignment='top', transform=axs.transAxes, fontproperties=font.get('i'))
		axs.text(0.87, 0.95, '= ' + ('%.3f' % r), horizontalalignment='left', verticalalignment='top', transform=axs.transAxes, fontproperties=font.get('n'))
	plt.setp(axs.get_xticklabels(), rotation=rotate_x_tics)
	plot_utils.set_fontproperties(font.get('b' if bold_x_tics else 'n'), axs.get_xticklabels())
	plot_utils.set_fontproperties(font.get('n'), axs.get_yticklabels())
	if hide_x_ticks: axs.xaxis.set_major_locator(ticker.NullLocator())
	if hide_y_ticks: axs.yaxis.set_major_locator(ticker.NullLocator())
	if despine or style in ['despine_ticks', 'whitegrid_ticks']: sb.despine(top=True, right=True)
	
	#Legend is a little tricky if the option is to be outside of the plot
	artists = []
	if show_legend:
		legend_out_kwargs = dict(bbox_to_anchor=(1, 1), loc=2, borderpad=legend_out_pad) if legend_out else None
		legend = axs.legend(**plot_utils.merged_kwargs(dict(prop=font.get('n')), legend_out_kwargs, legend_kwargs))
		artists.append(legend)		
				
	plt.tight_layout()
	if output is not None:
		plt.savefig(output, format=out_format, dpi=dpi, additional_artists=artists, bbox_inches='tight', pad_inches=fig_padding)
	else:
		plt.show()
	plt.close()
	
def main():
	print '#', ' '.join(sys.argv)
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('data_descriptors', nargs='+', type=str, help='list of data filenames and 0-based columns in this format: 1) multiple files: filenameA?columnX filenameB?columnY or 2) single file: filenameA?columnX?columnY')
	parser.add_argument('--auto_labels', action='store_true', help='labels for the data will be parsed from the first line of the file which has to start with "#"')
	parser.add_argument('--hide_corr', action='store_true', help='do not calculate and plot Pearson correlation (r)')
	parser.add_argument('--fit_reg', action='store_true', help='regression line')
	parser.add_argument('--ci', type=int, help='confidence interval 0...100 or None')
	plot_utils.add_common_arguments(parser, 'despine_ticks')
	args = parser.parse_args()
	try:
		plot_scatter(x=None, y=None, data_descriptors=args.data_descriptors, linewidth=args.linewidth, y_tics=args.ytics, x_range=args.xrange, y_range=args.yrange, x_log=args.xlog, y_log=args.ylog, sep=args.sep, title=args.title, x_label=args.xlabel, y_label=args.ylabel, rotate_x_tics=args.rotate_xtics, bold_x_tics=args.bold_xtics, hide_x_tick_marks=args.hide_xtick_marks, hide_y_tick_marks=args.hide_ytick_marks, hide_x_ticks=args.hide_xticks, hide_y_ticks=args.hide_yticks, auto_labels=args.auto_labels, label=args.label, show_corr=not args.hide_corr, show_legend=not args.hide_legend, legend_out=args.legend_out, legend_out_pad=args.legend_out_pad, fit_reg=args.fit_reg, ci=args.ci, despine=args.despine, style=args.style, fontsize=args.fontsize, colours=args.colours, palette=args.palette, reverse_palette=args.reverse_palette, ncolours=args.ncolours, figsize=args.figsize, fig_padding=args.fig_padding, dpi=args.dpi, output=args.output, out_format=args.format, kwargs=plot_utils.str_to_kwargs(args.kwargs) if args.kwargs is not None else None)
	except PlotsError as e:
		parser.error(e.message)
	
if __name__ == '__main__':
	main()
