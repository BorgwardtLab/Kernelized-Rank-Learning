#!/usr/bin/env python

import matplotlib as mpl
import os
import seaborn as sb
import numpy as np
import pandas as pd

GRID_STYLES = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks', 'white_ticks', 'whitegrid_ticks', 'despine_ticks']
ESTIMATORS = {'mean': np.mean, 'median': np.median, 'max': max, 'min': min, 'std': np.std, 'var': np.var, 'sum': sum}

FONT_DIR = os.path.join('fplots', 'fonts')
HELVETICA = os.path.join(FONT_DIR, 'Helvetica.ttf')
HELVETICA_BOLD = os.path.join(FONT_DIR, 'HelveticaBold.ttf')
HELVETICA_ITALIC = os.path.join(FONT_DIR, 'HelveticaOblique.ttf')
HELVETICA_BOLD_ITALIC = os.path.join(FONT_DIR, 'HelveticaBoldOblique.ttf')

RED = '#e41a1c'
GRAY = '#999999'
BLUE = '#377eb8'
PURPLE = '#984ea3'
GREEN = '#4daf4a'
ORANGE = '#ff7f00'
YELLOW = '#ffd92f'
BROWN = '#a65628'
PINK = '#d58a94'
LIGHT_GREEN = '#9dc100'
LIGHT_BLUE = '#95d0fc'
DARK_RED = '#840000'
BEIGE = '#d1b26f'
CYAN = '#13eac9'
OLIVE = '#6e750e'

MY_PALETTE = [RED, GRAY, BLUE, ORANGE, GREEN, PURPLE, BROWN, YELLOW, PINK, GRAY, CYAN, LIGHT_GREEN, LIGHT_BLUE, DARK_RED, BEIGE, OLIVE]

class PlotsError(Exception):
	def __init__(self, message):
		self.message = message

def read_table(filename, usecols=None, names=None, converters=None, sep=None, comment='#', header=None, na_filter=False):
	try:
		dataframe = pd.read_table(filename, usecols=usecols, names=names, converters=converters, sep=sep, comment=comment, header=header, na_filter=na_filter)
	except ValueError as e:
		raise PlotsError(message=e.message + ' (Do you have the correct separator, sep: "{0}"?)'.format(sep if sep != '\t' else '\\t'))
	
	return dataframe

def init_plot_style(style, fontsize, colours, palette, reverse_palette=False, ncolours=None, hide_x_tick_marks=False, hide_y_tick_marks=False):
	mpl.rc('pdf', fonttype=42)
	set_latex_fonts(sf_bold_italic_font='Helvetica')
	
	'''
	# other ways of changing fonts and sizes, but this would not work with a hacked version of Helvetica
	sb.set_style(style, rc={'font.sans-serif': ['Helvetica', 'Arial', 'Liberation Sans']})
	sb.set(rc={'axes.labelsize': fontsize, 'axes.titlesize': fontsize, 'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize, 'legend.fontsize': fontsize})
	'''
	
	if style == 'despine_ticks': sb.set_style('ticks')
	elif style == 'whitegrid_ticks':
		sb.set_style('whitegrid', {'linewidth': 1.25, 'axes.edgecolor': '0.15',
								   'xtick.minor.size': 2 if not hide_x_tick_marks else 0,
								   'xtick.major.size': 4 if not hide_x_tick_marks else 0,
								   'ytick.minor.size': 2 if not hide_y_tick_marks else 0,
								   'ytick.major.size': 4 if not hide_y_tick_marks else 0})
	elif style == 'white_ticks':
		sb.set_style('white', {'linewidth': 1.25, 'axes.edgecolor': '0.15',
								   'xtick.minor.size': 2 if not hide_x_tick_marks else 0,
								   'xtick.major.size': 4 if not hide_x_tick_marks else 0,
								   'ytick.minor.size': 2 if not hide_y_tick_marks else 0,
								   'ytick.major.size': 4 if not hide_y_tick_marks else 0})
	else: sb.set_style(style)
	colour_palette = sb.color_palette(colours if colours is not None else (palette if palette is not None else MY_PALETTE))
	sb.set_palette(colour_palette if not reverse_palette else (colour_palette[:ncolours])[::-1], n_colors=ncolours)
	
	#all of this trouble just to have Helvetica on any computer, given a TTF Helvetica file:
	if os.path.exists(HELVETICA) and os.path.exists(HELVETICA_BOLD) and os.path.exists(HELVETICA_ITALIC) and os.path.exists(HELVETICA_BOLD_ITALIC):
		normal = mpl.font_manager.FontProperties(fname=HELVETICA)
		bold = mpl.font_manager.FontProperties(fname=HELVETICA_BOLD)
		italic = mpl.font_manager.FontProperties(fname=HELVETICA_ITALIC)
		bold_italic = mpl.font_manager.FontProperties(fname=HELVETICA_BOLD_ITALIC)
	else:
		import warnings
		warnings.warn('Warning: could not locate Helvetica, using default font!')
		normal = mpl.font_manager.FontProperties()
		bold = mpl.font_manager.FontProperties()
		italic = mpl.font_manager.FontProperties()
		bold_italic = mpl.font_manager.FontProperties()

	normal.set_size(fontsize); normal.set_weight('normal'); normal.set_style('normal')
	bold.set_size(fontsize); bold.set_weight('bold'); bold.set_style('normal')
	italic.set_size(fontsize); italic.set_weight('normal'); italic.set_style('italic')
	bold_italic.set_size(fontsize); bold_italic.set_weight('bold'); bold_italic.set_style('italic')
	
	return {'n': normal, 'b': bold, 'i': italic, 'bi': bold_italic}
			
def set_latex_fonts(sf_bold_italic_font=None, rm_font=None, tt_font=None):
	#Not sure if this function does the correct thing
	#LaTeX fonts were not tested
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

def set_fontproperties(font, *elements):
	for es in elements:
		for e in es:
			e.set_fontproperties(font)

def fix_minus(value):
	try:
		float(value)
		return str(value).replace('-', '$-$')
	except Exception:
		return value

def merged_kwargs(*kwargs_dictionaries):
	merged = {}
	for d in kwargs_dictionaries:
		if d is not None:
			for key in d:
				merged[key] = d[key]
	return merged

def str_to_kwargs(args):
	kwargs = {}
	if args is not None:
		for arg in args:
			if arg.count('=') != 1: raise PlotsError('kwargs must take form name=value')
			name, value = arg.split('=')
			#is it int or float?
			try: value = float(value)
			except Exception: pass
			#is it a boolean?
			if value == 'True': value=True
			elif value == 'False': value=False
			kwargs[name] = value
	return kwargs

def add_common_arguments(parser, default_style):
	if default_style not in GRID_STYLES: raise ValueError
	parser.add_argument('-lw', '--linewidth', type=float, help='the width of the line', default=2)
	parser.add_argument('--hide_legend', action='store_true', help='hide the legend')
	parser.add_argument('--legend_out', action='store_true', help='legend outside of the plot')
	parser.add_argument('--legend_out_pad', type=float, help='border pad around legend if outside of the plot')
	parser.add_argument('--title', type=str, help='plot title')
	parser.add_argument('--label', type=str, help='plot label')
	parser.add_argument('--sep', type=str, help='delimiter in the data files', default='\t')
	parser.add_argument('--xrange', nargs='+', type=float, help='range for the x-axis specified as "min max"')
	parser.add_argument('--yrange', nargs='+', type=float, help='range for the y-axis specified as "min max"')
	parser.add_argument('--ytics', nargs='+', type=float, help='specific ticx on y axis')
	parser.add_argument('--xlabel', type=str, help='x-axis label')
	parser.add_argument('--ylabel', type=str, help='y-axis label')
	parser.add_argument('--xlog', action='store_true', help='log scale for x-axis')
	parser.add_argument('--ylog', action='store_true', help='log scale for y-axis')
	parser.add_argument('--rotate_xtics', type=float, help='Rotate x-axis tics by this many degrees (e.g., 45)')
	parser.add_argument('--bold_xtics', action='store_true', help='plot xtics using bold font')
	parser.add_argument('--hide_xtick_marks', action='store_true', help='hide xtics marks')
	parser.add_argument('--hide_ytick_marks', action='store_true', help='hide ytics marks')
	parser.add_argument('--hide_xticks', action='store_true', help='hide xtics')
	parser.add_argument('--hide_yticks', action='store_true', help='hide ytics')
	parser.add_argument('--style', choices=GRID_STYLES, help='sets the grid and axis style', default=default_style)
	parser.add_argument('--despine', action='store_true', help='removes the mirrored top and right axis')
	parser.add_argument('--colours', nargs='+', type=str, help='list of colours startin with "#" (mutually exclusive with --palette)')
	parser.add_argument('--palette', help='name of a ColorBrewer palette (mutually exclusive with --colours)', choices=['Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3', 'YlGn', 'YlGnBu', 'GnBu', 'BuGn', 'PuBuGn', 'PuBu', 'BuPu', 'RdPu', 'PuRd', 'OrRd', 'YlOrRd', 'YlOrBr', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'Greys', 'PuOr', 'BrBG', 'PRGn', 'PiYG', 'RdBu', 'RdGy', 'RdYlBu', 'Spectral', 'RdYlGn'])
	parser.add_argument('--reverse_palette', action='store_true', help='use colours of the current palette in a reverse order')
	parser.add_argument('--ncolours', type=int, help='how many colours from the specified palette to repeat (cannot be set if --colours are manually specified)')
	parser.add_argument('--dpi', type=int, help='output resolution')
	parser.add_argument('--figsize', nargs='+', type=float, help='figure size specified as "width height"')
	parser.add_argument('--fig_padding', type=float, help='figure padding size')
	parser.add_argument('--fontsize', type=int, help='font size', default=16)
	parser.add_argument('--legend_kwargs', nargs='+', type=str, help='**kwargs for the legend, e.g. "ncol=2 markerscale=3 ..."; see http://matplotlib.org/api/legend_api.html')
	parser.add_argument('--kwargs', nargs='+', type=str, help='**kwargs for all plots, e.g. "linewidth=5 ..."')
	parser.add_argument('--format', choices=['pdf', 'eps', 'ps', 'svg', 'png'], help='if not provided, the output format is determined from the extension of --output')
	parser.add_argument('-o', '--output', type=str, help='output filename')
	
