
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure


modules = ['add_or_sub', 'add_sub_multiple', 'div', 'mixed', 'mul', 'mul_div_multiple']
difficulties = ['train_easy', 'val_easy', 'medium', 'hard', 'extrapolate']
metrics = ['answer acc', 'char acc', 'loss']
metric_title = ['Answer Accuracy', 'Character Accuracy', 'Cross Entropy Loss']
fig, ax = plt.subplots(1,3, figsize=(16, 6), dpi=80)

for idx, metric in enumerate(metrics):

	scores = {}
	data = {}
	for module in modules:
		with open("./checkpoints/final_transformer_06-04-2020_18-55-27/best_val_eval/" + module + ".txt", 'rb') as log_file:
			data[module] = pickle.load(log_file)


	for module in modules:
		scores[module] = []
		for lvl in difficulties:
			scores[module].append(data[module][lvl][metric])

	labels = difficulties.copy()







	x = np.arange(len(difficulties))  # the label locations
	width = 0.1  # the width of the bars

	rects = []
	for i, module in enumerate(modules):
		rects.append(ax[idx].bar(x - width*3 + width*i, scores[module], width, label=module))


	# Add some text for labels, title and custom x-ax is tick labels, etc.
	ax[idx].set_ylabel(metric
		_title[idx])
	ax[idx].set_title("Final Transformer " + metric_title[idx])
	ax[idx].set_xticks(x)
	ax[idx].set_xticklabels(labels)
	ax[idx].legend() # change legend positon here
	# tried to move legend but it didn't look good
	# if metric == 'char acc':
	# 	ax[idx].legend(loc='lower left')
	# else:
	# 	ax[idx].legend() # change legend positon here

	# fix scale
	if metric == 'answer acc' or metric == 'char acc':
		ax[idx].set_ylim([0,1.0])
	else:
		ax[idx].set_ylim([0,6])




	def autolabel(rects,idx):
	    """Attach a text label above each bar in *rects*, displaying its height."""
	    for rect in rects:
	        height = rect.get_height()
	        ax[idx].annotate('{}'.format(height),
	                    xy=(rect.get_x() + rect.get_width() / 2, height),
	                    xytext=(0, 3),  # 3 points vertical offset
	                    textcoords="offset points",
	                    ha='center', va='bottom')


	# for i in rects:
	# 	autolabel(i, idx)
	# Show the major grid lines with dark grey lines
	ax[idx].grid(b=True, which='major', color='#666666', linestyle='-')

	# Show the minor grid lines with very faint and almost transparent grey lines
	ax[idx].minorticks_on()
	ax[idx].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
fig.tight_layout()

plt.show()
