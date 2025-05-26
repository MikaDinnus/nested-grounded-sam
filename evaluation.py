import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_excel('validation_values.xlsx')
file['TYPE'] = file['TYPE'].str.strip()

pivot_precision = file.pivot(index='CODE', columns='TYPE', values='PRECISION')
pivot_recall = file.pivot(index='CODE', columns='TYPE', values='RECALL')
pivot_iou = file.pivot(index='CODE', columns='TYPE', values='MEAN IOU OF BBOX')

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

pivot_precision.plot(kind='bar', ax=axes[0], width=0.8)
axes[0].set_title('Precision')
axes[0].set_xlabel('CODE')
axes[0].set_ylabel('Wert')
axes[0].legend(title='TYPE')

pivot_recall.plot(kind='bar', ax=axes[1], width=0.8)
axes[1].set_title('Recall')
axes[1].set_xlabel('CODE')
axes[1].set_ylabel('')
axes[1].legend(title='TYPE')

pivot_iou.plot(kind='bar', ax=axes[2], width=0.8)
axes[2].set_title('Mean IoU of BBox')
axes[2].set_xlabel('CODE')
axes[2].set_ylabel('')
axes[2].legend(title='TYPE')

plt.tight_layout()
plt.show()
