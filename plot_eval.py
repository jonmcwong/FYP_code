
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

modules = ['add_or_sub', 'add_sub_multiple', 'div', 'mixed', 'mul', 'mul_div_multiple']
difficulties = ['train_easy', 'medium', 'hard', 'interpolate', 'extrapolate']
metrics = ['answer acc', 'char acc', 'loss']
metric_title = ['Answer Accuracy', 'Character Accuracy', 'Cross Entropy Loss']
fig, ax = plt.subplots(1,3)

for idx, metric in enumerate(metrics):

	scores = {}
	data = {}
	for module in modules:
		with open("./checkpoints/transformer_1024_arithmetic_traineasy_lr_6e-6_05-24-2020_18-12-55/best_val_eval/" + module + ".txt", 'rb') as log_file:
			data[module] = pickle.load(log_file)


	for module in modules:
		scores[module] = []
		for lvl in difficulties:
			scores[module].append(data[module][lvl][metric])

	labels = difficulties.copy()
	labels[0] = 'easy'







	x = np.arange(len(difficulties))  # the label locations
	width = 0.1  # the width of the bars

	rects = []
	for i, module in enumerate(modules):
		rects.append(ax[idx].bar(x - width*3 + width*i, scores[module], width, label=module))


	# Add some text for labels, title and custom x-ax is tick labels, etc.
	ax[idx].set_ylabel(metric_title[idx])
	ax[idx].set_title(metric_title[idx] +' By Module')
	ax[idx].set_xticks(x)
	ax[idx].set_xticklabels(labels)
	ax[idx].legend()


	def autolabel(rects):
	    """Attach a text label above each bar in *rects*, displaying its height."""
	    for rect in rects:
	        height = rect.get_height()
	        ax.annotate('{}'.format(height),
	                    xy=(rect.get_x() + rect.get_width() / 2, height),
	                    xytext=(0, 3),  # 3 points vertical offset
	                    textcoords="offset points",
	                    ha='center', va='bottom')


	# for i in rects:
	# 	autolabel(i)

fig.tight_layout()

plt.show()
