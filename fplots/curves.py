#!/usr/bin/env python

import sys
import argparse
import matplotlib as mpl
import os
from fcommons import ml_utils
import plot_utils
if os.path.isfile('.pyplot_noX'): mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sb
from plot_utils import PlotsError

def set_latex_fonts(sf_bold_italic_font=None, rm_font=None, tt_font=None):
	if sf_bold_italic_font is not None or rm_font is not None or tt_font is not None:
		mpl.rcParams['mathtext.fontset'] = 'custom'
		if sf_bold_italic_font is not None:
			mpl.rcParams['mathtext.sf'] = sf_bold_italic_font #\mathsf{}
			mpl.rcParams['mathtext.cal'] = sf_bold_italic_font + ':bold' #\mathcal{}
			mpl.rcParams['mathtext.it'] = sf_bold_italic_font + ':italic' #\mathit{}
			mpl.rcParams['mathtext.bf'] = sf_bold_italic_font + ':italic:bold' #\mathbf{}
		if rm_font is not None:
			mpl.rcParams['mathtext.rm'] = rm_font #\mathrm{}
		if tt_font is not None:
			mpl.rcParams['mathtext.tt'] = tt_font #\mathtt{}

def make_curve_tuples(df, columns):
	data, x, y = [], None, None
	for i in range(0, len(columns), 2):
		x = df[columns[i]].tolist()
		if i < len(columns) - 1:
			y = df[columns[i+1]].tolist()
			data.append((x, y))
			x, y = None, None
	return data, x, y
			
def parse_data(data_descriptors, sep='\t'):
	data, x, y = [], None, None
	for data_descriptor in data_descriptors:
		dd_split = data_descriptor.split('?')
		data_filename = dd_split[0]
		columns = map(int, dd_split[1:])
		df = plot_utils.read_table(data_filename, usecols=columns, sep=sep)
		if x is None:
			subdata, x, y = make_curve_tuples(df, columns)
			data.extend(subdata)
		else:
			y = df[columns[0]].tolist()
			data.append((x, y))
			subdata, x, y = make_curve_tuples(df, columns[1:])
			data.extend(subdata)
			
	if x is not None: PlotsError(message='odd number of columns to plot')
	
	return data

def read_labels(data_filename, column, sep='\t'):
	with open(data_filename, 'r') as f:
		first_line = f.readline()
	if first_line[0] != '#': raise ValueError
	return first_line.strip('\n').split(sep)[column]

# The same as plot_curves() but xlabel and ylabel are set for convenience
def plot_roc(data, data_descriptors=None, labels=None, linewidth=2, \
			x_range=None, y_range=None, sep='\t', \
			title=None, rotate_x_tics=None, bold_x_tics=False, \
			hide_x_tick_marks=False, hide_y_tick_marks=False,\
			label=None, show_legend=True, legend_out=False, legend_out_pad=None, show_auc=True, despine=False, \
			style='despine_ticks', font=None, fontsize=16, colours=None, palette=None, reverse_palette=False, ncolours=None, \
			figsize=None, fig_padding=0.1, dpi=None, output=None, out_format=None, legend_kwargs=None, kwargs=None):
	
	plot_curves(x_label='False positive rate', y_label='True positive rate',\
			data=data, data_descriptors=data_descriptors, labels=labels, linewidth=linewidth, \
			x_range=x_range, y_range=y_range, sep=sep, \
			title=title, rotate_x_tics=rotate_x_tics, bold_x_tics=bold_x_tics, \
			hide_x_tick_marks=hide_x_tick_marks, hide_y_tick_marks=hide_y_tick_marks,\
			label=label, show_legend=show_legend, legend_out=legend_out, legend_out_pad=legend_out_pad, show_auc=show_auc, despine=despine, \
			style=style, font=font, fontsize=fontsize, colours=colours, palette=palette, reverse_palette=reverse_palette, ncolours=ncolours, \
			figsize=figsize, fig_padding=fig_padding, dpi=dpi, output=output, out_format=out_format, legend_kwargs=legend_kwargs, kwargs=kwargs)

# The same as plot_curves() but xlabel and ylabel are set for convenience	
def plot_prc(data, data_descriptors=None, labels=None, linewidth=2, \
			x_range=None, y_range=None, sep='\t', \
			title=None, rotate_x_tics=None, bold_x_tics=False, \
			hide_x_tick_marks=False, hide_y_tick_marks=False,\
			label=None, show_legend=True, legend_out=False, legend_out_pad=None, show_auc=True, despine=False, \
			style='despine_ticks', font=None, fontsize=16, colours=None, palette=None, reverse_palette=False, ncolours=None, \
			figsize=None, fig_padding=0.1, dpi=None, output=None, out_format=None, legend_kwargs=None, kwargs=None):

	plot_curves(x_label='Recall', y_label='Precision',\
			data=data, data_descriptors=data_descriptors, labels=labels, linewidth=linewidth, \
			x_range=x_range, y_range=y_range, sep=sep, \
			title=title, rotate_x_tics=rotate_x_tics, bold_x_tics=bold_x_tics, \
			hide_x_tick_marks=hide_x_tick_marks, hide_y_tick_marks=hide_y_tick_marks,\
			label=label, show_legend=show_legend, legend_out=legend_out, legend_out_pad=legend_out_pad, show_auc=show_auc, despine=despine, \
			style=style, font=font, fontsize=fontsize, colours=colours, palette=palette, reverse_palette=reverse_palette, ncolours=ncolours, \
			figsize=figsize, fig_padding=fig_padding, dpi=dpi, output=output, out_format=out_format, legend_kwargs=legend_kwargs, kwargs=kwargs)
	
def plot_curves(data, data_descriptors=None, labels=None, linewidth=2, y_tics=None, \
			x_range=None, y_range=None, x_log=False, y_log=False, sep='\t',\
			title=None, x_label=None, y_label=None, rotate_x_tics=None, bold_x_tics=False,\
			hide_x_tick_marks=False, hide_y_tick_marks=False,\
			hide_x_ticks=False, hide_y_ticks=False,\
			label=None, show_legend=True, legend_out=False, legend_out_pad=None, show_auc=True, despine=False,\
			style='despine_ticks', font=None, fontsize=16, colours=None, palette=None, reverse_palette=False, ncolours=None,\
			figsize=None, fig_padding=0.1, dpi=None, output=None, out_format=None, legend_kwargs=None, kwargs=None):
	'''
	Parameters
	----------
	data : a list of tuples of lists aka a list of curves
	[([x11, x12, ...], [y11, y12, ...]), ([x21, x22, ...], [y21, y22, ...]), ...] 
	'''
	if data is not None:
		for x, y in data:
			if len(x) != len(y):
				PlotsError(message='unpaired samples detected: x size: ' + str(len(x)) + ' y size: ' + str(len(y)))
		if data_descriptors is not None: raise PlotsError(message='You can specify only one of the mutually exclusive arguments: "x" and "y" or "data_descriptors"')
	else:
		if data_descriptors is None: raise PlotsError(message='You must specify "data" or "data_descriptors"')
		total_lists = 0
		for data_descriptor in data_descriptors:
			total_lists += data_descriptor.count('?')
		if total_lists % 2  != 0:
			raise PlotsError(message='you need to specify an even number of columns ' + str(total_lists))
	
	if not show_legend and (legend_out or legend_kwargs != None): raise PlotsError(message='If you hide the legend (--hide_legend or show_legend=False), you cannot set it outside (legend_out) or set it properties (legend_kwargs)')
	for x_y_range in [(x_range, 'xrange'), (y_range, 'yrange')]:
		if x_y_range[0] is not None and (len(x_y_range[0]) != 2 or x_y_range[0][0] >= x_y_range[0][1]): raise PlotsError(message='You need to provide exactly two numbers to set %s: "min max"' % x_y_range[1])
	if colours is not None:
		if palette is not None: raise PlotsError(message='You can specify only one of the mutually exclusive arguments: "colours" or "palette"')
	if figsize is not None and len(figsize) != 2: raise PlotsError(message='You need to provide exactly two numbers to set figure size: "width height"')
	
	if data is None:
		data = parse_data(data_descriptors, sep=sep)
	if labels is not None:
		if len(data) != len(labels): PlotsError(message='wrong number of labels: ' + str(len(data)) + ', ' + str(len(labels)))
		
	font = plot_utils.init_plot_style(style, fontsize, colours, palette, reverse_palette, ncolours, hide_x_tick_marks=hide_x_tick_marks, hide_y_tick_marks=hide_y_tick_marks)
	fig = plt.figure()
	if figsize is not None: fig.set_figwidth(figsize[0]); fig.set_figheight(figsize[1])
	if dpi is not None: fig.set_dpi(dpi)
	axs = plt.gca()
	
	for i, curve in enumerate(data):
		legend_label = labels[i] if labels is not None else ('curve ' + str(i + 1))
		if show_auc:
			auc = ml_utils.calc_AUC(zip(curve[0], curve[1]))
			legend_label += ' (AUC = ' + ('%.3f' % auc) + ')'
		plt.plot(curve[0], curve[1], **plot_utils.merged_kwargs({'label': legend_label, 'linewidth': linewidth}, kwargs))
	if x_range is not None: axs.set_xlim(x_range[0], x_range[1])
	if y_range is not None: axs.set_ylim(y_range[0], y_range[1])
	if y_tics is not None: axs.set_yticks(y_tics)
	if x_log: axs.set_xscale('log')
	if y_log: axs.set_yscale('log')
	if x_label is not None: axs.set_xlabel(x_label, labelpad=15, fontproperties=font.get('b'))
	if y_label is not None: axs.set_ylabel(y_label, labelpad=15, fontproperties=font.get('b'))
	if title is not None: axs.set_title(title, fontproperties=font.get('b'))
	if label is not None: axs.text(0.05, 0.95, label, horizontalalignment='left', verticalalignment='top', transform=axs.transAxes, fontproperties=font.get('n'))
	if show_legend: axs.legend(loc='lower right', prop=font.get('n'))
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
		legend = axs.legend(**plot_utils.merged_kwargs(dict(prop=font.get('n'), loc='lower right'), legend_out_kwargs, legend_kwargs))
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
	parser.add_argument('data_descriptors', nargs='+', type=str, help='list of data filenames and 0-based columns in this format: 1) multiple files: filenameA?columnX filenameB?columnY ... or 2) single file: filenameA?columnX?columnY?columnZ')
	parser.add_argument('--labels', nargs='+', type=str, help='labels for different curves')
	parser.add_argument('--hide_auc', action='store_true', help='do not calculate and show AUC in the legend')
	plot_utils.add_common_arguments(parser, 'despine_ticks')
	args = parser.parse_args()
	
	try:
		plot_curves(data=None, data_descriptors=args.data_descriptors, labels=args.labels, linewidth=args.linewidth, y_tics=args.ytics, x_range=args.xrange, y_range=args.yrange, x_log=args.xlog, y_log=args.ylog, sep=args.sep, title=args.title, x_label=args.xlabel, y_label=args.ylabel, rotate_x_tics=args.rotate_xtics, bold_x_tics=args.bold_xtics, hide_x_tick_marks=args.hide_xtick_marks, hide_y_tick_marks=args.hide_ytick_marks, hide_x_ticks=args.hide_xticks, hide_y_ticks=args.hide_yticks, label=args.label, show_legend=not args.hide_legend, legend_out=args.legend_out, legend_out_pad=args.legend_out_pad, show_auc=not args.hide_auc, despine=args.despine, style=args.style, fontsize=args.fontsize, colours=args.colours, palette=args.palette, reverse_palette=args.reverse_palette, ncolours=args.ncolours, figsize=args.figsize, fig_padding=args.fig_padding, dpi=args.dpi, output=args.output, out_format=args.format, kwargs=plot_utils.str_to_kwargs(args.kwargs) if args.kwargs is not None else None)
	except PlotsError as e:
		parser.error(e.message)
	
if __name__ == '__main__':
	main()
