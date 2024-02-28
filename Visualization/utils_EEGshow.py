import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def adjust_window(time, length, L, sfreq):
    time = int(time * sfreq)
    length = int(length * sfreq)
    start = time
    end = time + length

    if start < 0:
        start = 0
    elif end > L - 1:
        end = L - 1
    elif start > L - 1:
        start = L - 1 - length
        end = L - 1
    ll = (end - start) / sfreq

    return start, end, ll


def PlotEEGMontage(eeg_signal, time, length, label=None, pred=None, file_name=None, Sens=1.5, save_fig=None, sfreq=125):
    L, N = eeg_signal.shape
    # sfreq = 250

    start, end, ll = adjust_window(time, length, L, sfreq)

    columns = eeg_signal.columns

    data = eeg_signal.values
    x_index = np.linspace(time, time + ll, end - start)

    anchor_xaxis = np.arange(0, N * 100 * Sens, 100 * Sens)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)

    cmap = plt.get_cmap("Set2")

    colors = [cmap(i) for i in np.linspace(0, 1, 23)]
    if label is not None:

        label = label[(label['start'] < end) & (label['end'] > start)]
        label.loc[label['end'] > end, 'end'] = end
        label.loc[label['start'] < start, 'start'] = start

        for label_start, label_end, j, name in label.loc[:, ['start', 'end', '#Channel', 'label']].values:
            rect = plt.Rectangle((x_index[label_start - start], (N - j - 1 - 0.5) * 100 * Sens),
                                 (label_end - label_start) / sfreq, 100 * Sens * 0.8)
            rect.set(facecolor='gray', alpha=0.5)
            ax.add_patch(rect)
            ax.text(x_index[label_start - start], (N - j - 1) * 100 * Sens, name, color='black')

    if pred is not None:

        pred = pred[(pred['start'] < end) & (pred['end'] > start)]
        pred.loc[pred['end'] > end, 'end'] = end
        pred.loc[pred['start'] < start, 'start'] = start

        if 'label' not in pred.columns:
            for pred_start, pred_end, j in pred.loc[:, ['start', 'end', '#Channel']].values:
                line = plt.plot(x_index[pred_start - start:pred_end - start],
                                data[pred_start:pred_end, j] + anchor_xaxis[N - j - 1],
                                color='red',
                                linewidth=5)
                # ax.text(x_index[pred_start - start], (N - j - 1.5) * 100 * Sens, name, color=pred_colors[k])

            # 创建自定义的 Line2D 对象
            lines = []
            lines.append(Line2D([], [], color='red', linestyle='-', label='Artifacts'))
        else:
            colors = ['black'] * 23
            pred_colors = ['white', 'red', 'skyblue', 'limegreen', 'pink', 'gold']
            linewidths = [1, 2, 3, 4, 5, 6]
            label_name = ['eyem', 'chew', 'shiv', 'musc', 'elpp/elec']
            pred = pred.sort_values('#label_S', ascending=False)
            for pred_start, pred_end, j, name, k in pred.loc[:,
                                                    ['start', 'end', '#Channel', 'label', '#label_S']].values:
                line = plt.plot(x_index[pred_start - start:pred_end - start],
                                data[pred_start:pred_end, j] + anchor_xaxis[N - j - 1],
                                color=pred_colors[k],
                                linewidth=linewidths[k])
                ax.text(x_index[pred_start - start], (N - j - 1.5) * 100 * Sens, name, color=pred_colors[k])

            # 创建自定义的 Line2D 对象
            lines = []
            for c, n in zip(pred_colors[1:], label_name):
                lines.append(Line2D([], [], color=c, linestyle='-', label=n))

        # 创建图例
        legend = plt.legend(handles=lines, loc='upper right')

        ax.add_artist(legend)

    for i in range(N):
        ax.plot(x_index, data[start:end, N - i - 1] + anchor_xaxis[i], color=colors[N - i - 1], linewidth=1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks(anchor_xaxis)
    ax.set_yticklabels(columns[::-1], rotation=0)
    ax.set_ylim([-100 * Sens, 100 * Sens * N])
    ax.set_xticks(list(np.arange(time, time + length + 1e-6, length / 5)))

    plt.title('[{}] {}s-{}s'.format(file_name, time, time + length))
    # plt.subplots_adjust(hspace=0.05)
    if save_fig is not None:
        path = "{}_{}_{}_inference.png".format(save_fig, file_name, time)
        # path = os.path.join('PlotDataset2', "{}-{}.png".format(save_fig, time))
        plt.savefig(path, dpi=500, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return None
