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
import pandas as pd
from plot_utils import PlotsError
import plot_utils

def make_xyhue_dataframe(x, y, hue):
	data_dict = {}
	if x is not None: data_dict['x'] = np.array(x)
	if y is not None: data_dict['y'] = np.array(y)
	if hue is not None: data_dict['hue'] = np.array(hue)
	return pd.DataFrame(data_dict)

def read_grid_as_xyhue(filename, grid_columns=None, x_tics=None, sep=None, comment='#'):
	if grid_columns is not None and x_tics is not None and len(grid_columns) != len(x_tics): raise ValueError
	x, y, hue = [], [], []
	with open(filename, 'r') as f:
		for line in f:
			if line[0] == comment: continue
			line_split = line.strip('\n').split(sep)
			if grid_columns is None: grid_columns = range(1,len(line_split))
			for i, column in enumerate(grid_columns):
				if column > len(line_split) - 1:
					raise PlotsError(message='Column index out of bounds (do you have the correct separator, sep: "{0}"?)'.format(sep if sep != '\t' else '\\t'))
				hue.append(line_split[0])
				x.append(x_tics[i] if x_tics is not None else column)
				y.append(float(line_split[column]))
	#if there is only a single hue category, ignore it (thanks to this, every x category will have a different colour, as expected)
	if len(set(hue)) == 1:
		hue = None
	return x, y, hue
	
def plot_distro(data, data_descriptor=None, ax=None, grid_columns=None, x_tics=None, y_tics=None, linewidth=2,\
				box_plot=False, violin_plot=False, bar_plot=False, point_plot=False, count_plot=False, swarm_plot=False, strip_plot=False,\
				y_range=None, x_log=False, y_log=False, sep=None, horizontal=False, order=None, hue_order=None,\
				box_outlier=5, box_whis=1.5, box_notch=False,\
				violin_scale='area', violin_inner='box', violin_cut=2, violin_split=False, violin_scale_hue=False,\
				estimator=plot_utils.ESTIMATORS['mean'], ci=95, capsize=0.2,\
				strip_jitter=True, points_colour=None,\
				point_markers='o', point_marker_size=2,\
				title=None, x_label=None, y_label=None, rotate_x_tics=None, bold_x_tics=False,\
				hide_x_tick_marks=False, hide_y_tick_marks=False,\
				hide_x_ticks=False, hide_y_ticks=False,\
				label=None, show_legend=True, legend_out=False, legend_out_pad=None,\
				despine=True, style='whitegrid_ticks', fontsize=16, colours=None, palette=None, reverse_palette=False, ncolours=None, figsize=None, fig_padding=0.1, dpi=None, output=None, out_format=None,\
				box_kwargs=None, violin_kwargs=None, bar_kwargs=None, point_kwargs=None, count_kwargs=None, swarm_kwargs=None, strip_kwargs=None, legend_kwargs=None, kwargs=None,
				color=None):
	'''
	Parameters
	----------
	data : pandas.DataFrame with indexes 'x', 'y', 'hue'
	or
	data : 3-tuple of lists x, y, and hue ([...], [...], [...]), thus for a single plot data=(None, [...], None)
	'''
	
	if not (box_plot or violin_plot or bar_plot or point_plot or count_plot or swarm_plot or strip_plot): raise PlotsError(message='Specify a plot to plot: box or violin or bar or count or swarm or strip')
	if count_plot and (box_plot or violin_plot or bar_plot or point_plot or swarm_plot or strip_plot): raise PlotsError(message='Count plot cannot be combined with any other plot')
	if point_plot and (box_plot or violin_plot or bar_plot or count_plot or swarm_plot or strip_plot): raise PlotsError(message='Point plot cannot be combined with any other plot')
	
	# PARSED DATA IS SUPPLIED AS pd.DataFrame({'x':[] , 'y':[] , 'hue':[] }) OR ([...], [...], [...])
	if data is not None:
		if data_descriptor is not None: raise PlotsError(message='You can specify only one of the mutually exclusive arguments: "data" or "data_descriptor"')
		if grid_columns is not None: raise PlotsError(message='The grid_columns option is only used when the data is supplied as a filename')
		# pd.DataFrame({'x':[] , 'y':[] , 'hue':[] }) 
		if isinstance(data, pd.DataFrame) and (('y' not in data and not count_plot) or ('x' not in data and count_plot)):
			raise PlotsError(message='The dataframe has to have a "y" column, optionally also "x" and "hue" columns' if not count_plot else 'The dataframe has to have a "x" column, optionally also a "hue" column')
		# ([...], [...], [...])
		else:
			if len(data) != 3 or ((data[1] is None and not count_plot) or (data[0] is None and count_plot)):
				raise PlotsError(message='The data should be a pandas.DataFrame or a 3-tuple of lists (x, y, hue). ' + ('The y list cannot be None, x and hue are optional (can be None).' if not count_plot else 'The x list cannot be None, hue is optional (can be None), y is ignored (set it None).'))
			x, y, hue = data
			data = make_xyhue_dataframe(x, y, hue)
	
	# DATA IS SUPPLIED AS A FILENAME - THE FILE IS A THREE_COLUMN_FILE OR A GRID_LIKE_FILE
	else:
		if data_descriptor is None: raise PlotsError(message='You must specify "data" or "data_descriptor"')
		# GRID_LIKE_FILE
		if '?' not in data_descriptor:
			if grid_columns is None: raise PlotsError(message='You must specify columns for y, or x and y, or x, y and hue, or specify grid-columns: filename?y or filename?x?y or filename?x?y?hue or filename --grid_columns x1 x2 x3')
			if x_tics is None: raise PlotsError(message='When specifying --grid_columns, you have to specify also the labels for xtics (--xtics)')
			if len(x_tics) != len(grid_columns): raise PlotsError(message='The number of columns (--grid_columns) differ from the number of xtics (--xtics)')
			x, y, hue = read_grid_as_xyhue(data_descriptor, grid_columns, x_tics, sep=sep, comment='#')
			data = make_xyhue_dataframe(x, y, hue)
			x_tics = None #hack; they have been already assigned; this will prevent re-assigning again below
		# THREE_COLUMN_FILE
		else:
			if grid_columns is not None: raise PlotsError(message='When specifying columns using ?, you cannot specify --grid_columns.')
			data_descriptor_split = data_descriptor.split('?')
			filename = data_descriptor_split[0]
			if len(data_descriptor_split) == 2:
				columns_names = {'y': int(data_descriptor_split[1])} if not count_plot else {'x': int(data_descriptor_split[1])}
			elif len(data_descriptor_split) == 3:
				columns_names = {'x': int(data_descriptor_split[1]), 'y': int(data_descriptor_split[2])} if not count_plot else {'x': int(data_descriptor_split[1]), 'hue': int(data_descriptor_split[2])}
			elif len(data_descriptor_split) == 4:
				if count_plot: raise PlotsError(message='For count plot, you can only specify x and hue')
				columns_names = {'x': int(data_descriptor_split[1]), 'y': int(data_descriptor_split[2]), 'hue': int(data_descriptor_split[3])}
			else:
				raise PlotsError(message='You can specify only up to 3 columns for x, y and hue: filename or filename?y or filename?x?y or filename?x?y?hue')
			names, columns = zip(*sorted(columns_names.items(), key=lambda x: x[1]))
			data = plot_utils.read_table(filename, usecols=columns, names=names, sep=sep)

	if x_tics is not None:
		if 'x' in data: raise PlotsError(message='You specified the x-categories in your data, thus you cannot use the xtics option')
		elif len(x_tics) != 1: raise PlotsError(message='You can specify only one x-category using xtics (unless you use a grid-like file input)')
		else: data['x'] = np.array([x_tics[0]] * len(data['y']))
	
	if not show_legend and (legend_out or legend_kwargs != None): raise PlotsError(message='If you hide the legend (--hide_legend or show_legend=False), you cannot set it outside (legend_out) or set it properties (legend_kwargs)')
	if y_range is not None and (len(y_range) != 2 or y_range[0] >= y_range[1]): raise PlotsError(message='You need to provide exactly two numbers to set yrange: "min max"')
	for my_order, variable in ((order, 'x'), (hue_order, 'hue')):
		if my_order is not None:
			if variable not in data: raise PlotsError(message='You specified order for %s but your data does not contain %s' % (variable, variable))
			set_variable = set(data[variable])
			if len(my_order) != len(set_variable) or set(my_order) != set_variable: raise PlotsError(message='The specified order does not match %s' % variable)
	if ci is not None and ci != 'std' and not callable(ci) and (ci < 0 or ci > 100): raise PlotsError(message='"ci" must be None or within 0 and 100')
	
	#if (swarm or strip) and 'hue' in data: raise PlotsError(message='Swarmplot is not supported when plotting plots with hue.')
	if colours is not None:
		if palette is not None: raise PlotsError(message='You can specify only one of the mutually exclusive arguments: "colours" or "palette"')
		if ncolours is not None: raise PlotsError(message='You cannot specify "ncolours" when you specified "colours"')
	if figsize is not None and len(figsize) != 2: raise PlotsError(message='You need to provide exactly two numbers to set figure size: "width height"')

	##
	
	font = plot_utils.init_plot_style(style, fontsize, colours, palette, reverse_palette, ncolours, hide_x_tick_marks=hide_x_tick_marks, hide_y_tick_marks=hide_y_tick_marks)
	fig = plt.figure()
	if figsize is not None: fig.set_figwidth(figsize[0]); fig.set_figheight(figsize[1])
	if dpi is not None: fig.set_dpi(dpi)
	
	default_kwargs = {'ax': ax, 'x': 'x' if 'x' in data else None, 'y': 'y' if 'y' in data else None, 'hue': 'hue' if 'hue' in data else None, 'data': data, 'orient': 'h' if horizontal else 'v', 'order': order, 'hue_order': hue_order, 'linewidth': linewidth}

	if box_plot:	axs = sb.boxplot(**plot_utils.merged_kwargs(default_kwargs, dict(fliersize=box_outlier, whis=box_whis, notch=box_notch, flierprops={'marker': 'o'}), kwargs, box_kwargs))
	if violin_plot:	axs = sb.violinplot(**plot_utils.merged_kwargs(default_kwargs, dict(scale=violin_scale, inner=violin_inner, cut=violin_cut, split=violin_split, scale_hue=violin_scale_hue), kwargs, violin_kwargs))
	if bar_plot:	axs = sb.barplot(**plot_utils.merged_kwargs(default_kwargs, dict(estimator=estimator, ci=ci, capsize=capsize), kwargs, bar_kwargs))
	if point_plot:  axs = sb.pointplot(**plot_utils.merged_kwargs(default_kwargs, dict(markers=point_markers, estimator=estimator, ci=ci, capsize=capsize), kwargs, point_kwargs))
	if count_plot:	axs = sb.countplot(**plot_utils.merged_kwargs(default_kwargs, kwargs, count_kwargs))
	if swarm_plot: axs = sb.swarmplot(**plot_utils.merged_kwargs(default_kwargs, dict(edgecolor='black', linewidth=1),  dict(facecolor=points_colour) if points_colour is not None else {}, kwargs, swarm_kwargs))
	if strip_plot: axs = sb.stripplot(**plot_utils.merged_kwargs(default_kwargs, dict(edgecolor='black', linewidth=1, jitter=strip_jitter, split=True), dict( facecolor=points_colour) if points_colour is not None else {}, kwargs, strip_kwargs))

	if point_plot:
		plt.setp(axs.collections, sizes=[point_marker_size])
		plt.setp(axs.lines, linewidth=linewidth)
	if y_range is not None: axs.set_ylim(y_range[0], y_range[1])
	axs.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
	if y_tics is not None: axs.set_yticks(y_tics)
	if x_log: axs.set_xscale('log')
	if y_log: axs.set_yscale('log')
	if x_label is not None: axs.set_xlabel(x_label, labelpad=8, fontproperties=font.get('b'))
	else: axs.set_xlabel('', labelpad=8)
	if y_label is not None: axs.set_ylabel(y_label, labelpad=10, fontproperties=font.get('b'))
	else: axs.set_ylabel('', labelpad=10)
	if title is not None:
		ttl = axs.set_title(title, fontproperties=font.get('b'))
		ttl.set_position([.5, 1.05])
	if label is not None:
		axs.text(label[1], label[2], label[0], horizontalalignment='left', verticalalignment='top', transform=axs.transAxes, fontproperties=font.get('b'))
	plt.setp(axs.get_xticklabels(), rotation=rotate_x_tics)
	plot_utils.set_fontproperties(font.get('b' if bold_x_tics else 'n'), axs.get_xticklabels())
	plot_utils.set_fontproperties(font.get('n'), axs.get_yticklabels())
	if hide_x_ticks: axs.xaxis.set_major_locator(ticker.NullLocator())
	if hide_y_ticks: axs.yaxis.set_major_locator(ticker.NullLocator())
	if despine or style in ['despine_ticks', 'whitegrid_ticks']: sb.despine(top=True, right=True)

	#Legend is a little tricky if the option is to be outside of the plot
	artists = []
	if show_legend:
		legend_handles, legend_labels = axs.get_legend_handles_labels()
		num_plots = sum([box_plot, violin_plot, bar_plot, point_plot, count_plot, swarm_plot, strip_plot])
		if num_plots != 1:
			legend_handles, legend_labels = legend_handles[:len(legend_handles)/num_plots], legend_labels[:len(legend_labels)/num_plots]
		legend_out_kwargs = dict(bbox_to_anchor=(1, 1), loc=2, borderpad=legend_out_pad) if legend_out else None
		legend_handles, legend_labels = legend_handles[::-1], legend_labels[::-1]
		legend = axs.legend(legend_handles, legend_labels, **plot_utils.merged_kwargs(dict(prop=font.get('n')), legend_out_kwargs, legend_kwargs))
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
	#DATA
	parser.add_argument('data_descriptor', type=str, help='data filename and 0-based columns in this format: 1) filename?y for a single plot, 2) filename?x?y (or filename?x for a countplot) for plots of y-values for each x category, 3) filename?x?y?hue (or filename?x?hue for a countplot) for plots of y-values for each x category and sub-category hue, 4) filename?_?y?hue for plots of y-values for each sub-category hue but with no x categories ("_"), or 5) filename --grid_columns x1 x2 x3 --x_ticks "One" "Two" "Three" --> for a grid-like files where hue is given in column 0, x categories are x1, x2, and x3')
	parser.add_argument('-g', '--grid_columns', nargs='+', type=int, help='grid-like (table-like) input where columns are the "x" categories and rows are the "hue" subcategories; the format is then changed so that the first column (index 0) is ALWAYS the labels of the hue subcategories, e.g. filename --grid_columns x1 x2 x3 --x_ticks "One" "Two" "Three"')
	parser.add_argument('-x', '--xtics', nargs='+', type=str, help='either specify 1 x-tic when supplying filename?y for a single plot or specify all x-tics when supplying filename --grid_columns (in this case xtics is mandatory)')
	parser.add_argument('--order', nargs='+', type=str, help='the order is given as a list of x categories')
	parser.add_argument('--hue_order', nargs='+', type=str, help='the order is given as a list of hue subcategories')
	
	#PLOTS
	parser.add_argument('-b', '--box', action='store_true', help='plot a boxplot')
	parser.add_argument('-v', '--violin', action='store_true', help='plot a violinplot')
	parser.add_argument('-r', '--bar', action='store_true', help='plot a barplot')
	parser.add_argument('-r', '--point', action='store_true', help='plot a pointplot')
	parser.add_argument('-c', '--count', action='store_true', help='plot a countplot')
	parser.add_argument('-s', '--swarm', action='store_true', help='plot a swarmplot')
	parser.add_argument('-p', '--strip', action='store_true', help='plot a stripplot')
	
	#KWARGS
	parser.add_argument('-bk', '--box_kwargs', nargs='+', type=str, help='**kwargs for the boxplot, e.g. "color=white edgecolor=gray linewidth=1"')
	parser.add_argument('-vk', '--violin_kwargs', nargs='+', type=str, help='**kwargs for the violinplot, e.g. "color=white edgecolor=gray linewidth=1"')
	parser.add_argument('-rk', '--bar_kwargs', nargs='+', type=str, help='**kwargs for the barplot, e.g. "color=white edgecolor=gray linewidth=1"')
	parser.add_argument('-rk', '--point_kwargs', nargs='+', type=str, help='**kwargs for the pointplot, e.g. "color=white edgecolor=gray linewidth=1"')
	parser.add_argument('-ck', '--count_kwargs', nargs='+', type=str, help='**kwargs for the countplot, e.g. "color=white edgecolor=gray linewidth=1"')
	parser.add_argument('-sk', '--swarm_kwargs', nargs='+', type=str, help='**kwargs for the swarmplot, e.g. "color=white edgecolor=gray linewidth=1"')
	parser.add_argument('-tk', '--strip_kwargs', nargs='+', type=str, help='**kwargs for the stripplot, e.g. "color=white edgecolor=gray linewidth=1"')
		
	#PLOT-SPECIFIC
	parser.add_argument('--box_outlier', type=float, help='Size of the markers used to indicate outlier observations', default=5)
	parser.add_argument('--box_whis', type=float, help='Proportion of the IQR past the low and high quartiles to extend the plot whiskers. Points outside this range will be identified as outliers', default=1.5)
	parser.add_argument('--box_notch', action='store_true', help='Whether to notch the box to indicate a confidence interval for the median. There are several other parameters that can control how the notches are drawn; see the plt.boxplot help for more information on them')
	parser.add_argument('--violin_scale', choices=['area', 'count', 'width'], help='The method used to scale the width of each violin. If area, each violin will have the same area. If count, the width of the violins will be scaled by the number of observations in that bin. If width, each violin will have the same width', default='area')
	parser.add_argument('--violin_inner', choices=['box', 'quartile', 'point', 'stick', 'None'], help='Representation of the datapoints in the violin interior. If box, draw a miniature boxplot. If quartiles, draw the quartiles of the distribution. If point or stick, show each underlying datapoint. Using None will draw unadorned violins', default='box')
	parser.add_argument('--violin_cut', action='store_true', help='limit the violin range within the range of the observed data')
	parser.add_argument('--violin_split', action='store_true', help='When using hue nesting with a variable that takes two levels, setting split to True will draw half of a violin for each level. This can make it easier to directly compare the distributions')
	parser.add_argument('--violin_scale_hue', action='store_true', help='When nesting violins using a hue variable, this parameter determines whether the scaling is computed within each level of the major grouping variable (scale_hue=True) or across all the violins on the plot (scale_hue=False)')
	parser.add_argument('--estimator', choices=['mean', 'median', 'max', 'min', 'std', 'var', 'sum'], help='method to estimate the statistic for the barplot and pointplot', default='mean')
	parser.add_argument('--point_markers', nargs='+', type=str, help='string or list (point markers for each of the hue levels)', default='o')
	parser.add_argument('--point_marker_size', type=float, help='size of the markers in the pointplot', default=2)
	parser.add_argument('--no_ci', action='store_true', help='do not plot the confidence interval')
	parser.add_argument('--ci', type=int, help='Size of confidence intervals to draw around estimated values (0...100). If None, no bootstrapping will be performed, and error bars will not be drawn', default=95)
	parser.add_argument('--capsize', type=float, help='Width of error line caps)', default=0.2)
	parser.add_argument('--strip_no_jitter', action='store_true', help='do not use jitter for the stripplot')
	parser.add_argument('--black_points', action='store_true', help='Black points for swarm and strip plots')
	parser.add_argument('--transp_points', action='store_true', help='Transparent points for swarm and strip plots')
	
	# THIS IS NOT WORKING FOR SOME UNKNOWN REASON
	parser.add_argument('--horizontal', action='store_true', help='place y-axis horizontally (swap x and y)')
	
	plot_utils.add_common_arguments(parser, 'whitegrid_ticks')
	args = parser.parse_args()
	if args.black_points and args.transp_points: parser.error('Cannot do black points and transparent points at the same time, choose one of the two options')
	try:
		plot_distro(data=None, data_descriptor=args.data_descriptor, grid_columns=args.grid_columns, x_tics=args.xtics, y_tics=args.ytics, linewidth=args.linewidth, box_plot=args.box, violin_plot=args.violin, bar_plot=args.bar, point_plot=args.point, count_plot=args.count, swarm_plot=args.swarm, strip_plot=args.strip, y_range=args.yrange, x_log=args.xlog, y_log=args.ylog, sep=args.sep, horizontal=args.horizontal, order=args.order, hue_order=args.hue_order, box_outlier=args.box_outlier, box_whis=args.box_whis, box_notch=args.box_notch, violin_scale=args.violin_scale, violin_inner=args.violin_inner if args.violin_inner != 'None' else None, violin_cut=0 if args.violin_cut else 2, violin_split=args.violin_split, violin_scale_hue=args.violin_scale_hue, estimator=plot_utils.ESTIMATORS[args.estimator], ci=args.ci if not args.no_ci else None, capsize=args.capsize, strip_jitter=not args.strip_no_jitter, points_colour='none' if args.transp_points else ('black' if args.black_points else None), point_markers=args.point_markers, point_marker_size=args.point_marker_size, title=args.title, x_label=args.xlabel, y_label=args.ylabel, rotate_x_tics=args.rotate_xtics, bold_x_tics=args.bold_xtics, hide_x_tick_marks=args.hide_xtick_marks, hide_y_tick_marks=args.hide_ytick_marks, hide_x_ticks=args.hide_xticks, hide_y_ticks=args.hide_yticks, label=args.label, show_legend=not args.hide_legend, legend_out=args.legend_out, legend_out_pad=args.legend_out_pad, despine=args.despine, style=args.style, fontsize=args.fontsize, colours=args.colours, palette=args.palette, reverse_palette=args.reverse_palette, ncolours=args.ncolours, figsize=args.figsize, fig_padding=args.fig_padding, dpi=args.dpi, output=args.output, out_format=args.format, box_kwargs=plot_utils.str_to_kwargs(args.box_kwargs) if args.box_kwargs is not None else None, violin_kwargs=plot_utils.str_to_kwargs(args.violin_kwargs) if args.violin_kwargs is not None else None, bar_kwargs=plot_utils.str_to_kwargs(args.bar_kwargs) if args.bar_kwargs is not None else None, point_kwargs=plot_utils.str_to_kwargs(args.point_kwargs) if args.point_kwargs is not None else None, count_kwargs=plot_utils.str_to_kwargs(args.count_kwargs) if args.count_kwargs is not None else None, swarm_kwargs=plot_utils.str_to_kwargs(args.swarm_kwargs) if args.swarm_kwargs is not None else None, strip_kwargs=plot_utils.str_to_kwargs(args.strip_kwargs) if args.strip_kwargs is not None else None, legend_kwargs=plot_utils.str_to_kwargs(args.legend_kwargs) if args.legend_kwargs is not None else None, kwargs=plot_utils.str_to_kwargs(args.kwargs) if args.kwargs is not None else None)
	except PlotsError as e:
		parser.error(e.message)
	
if __name__ == '__main__':
	main()
