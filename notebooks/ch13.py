import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf


def example_plot(xy, labels, a, b, title, filename=None):
    # Shape
    c_shape = ['bs', 'r^']

    # 1. Point
    for k, (point, label) in enumerate(zip(xy, labels),1):
        x,y = point
        plt.plot(x, y, c_shape[label[0]], mec='k', markersize=12)
        plt.text(x+0.02, y-0.01, '$P_{}$'.format(k), size=15, \
                verticalalignment='top', horizontalalignment='left')

    # 2. Decision line
    tmp = np.linspace(0,1,500)
    decision_line = a * tmp + b
    plt.plot(tmp, decision_line, 'k-', linewidth=3)

    # 3. Color
    x = np.array([0,1])
    y = a*x + b
    plt.fill_between(x,y,1, color='gray')

    plt.grid(linestyle='--', alpha=0.5)
    plt.xlim([-1.2,1.2])
    plt.ylim([-1.2,1.2])
    plt.title(title)
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()


def example_plot_only_line(xy, labels, a, b, title, filename=None):
    # Shape
    c_shape = ['bs', 'r^']

    # 1. Point
    for k, (point, label) in enumerate(zip(xy, labels),1):
        x,y = point
        plt.plot(x, y, c_shape[label[0]], mec='k', markersize=10)
        plt.text(x+0.02, y-0.01, '$P_{}$'.format(k), size=15, \
                verticalalignment='top', horizontalalignment='left')

    # 2. Decision line
    tmp = np.linspace(0,1,500)
    decision_line = a * tmp + b
    plt.plot(tmp, decision_line, 'k-', linewidth=3)

    plt.grid(linestyle='--', alpha=0.5)
    plt.xlim([-1.2,1.2])
    plt.ylim([-1.2,1.2])
    plt.title(title)
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

def plot_model(model, feature_labmda, xy, labels, xy2, labels2, title=''):
    from pandas import DataFrame
    xx, yy = np.meshgrid(np.linspace(-1.3,1.3, 400), np.linspace(-1.3,1.3, 400))
    input_xy = np.array([feature_labmda(xxval, yyval) for xxval, yyval in zip(xx.flatten(), yy.flatten())])
    prediction = model(input_xy).numpy()
    Z = prediction.reshape(xx.shape)
    df = DataFrame(dict(x=xy[:,0], y=xy[:,1], label=labels.flatten()))
    markers = {0:'bs', 1:'r^'}
    _, ax = plt.subplots(figsize=(7, 7))
    cs = ax.contourf(xx, yy, Z, 20, cmap=plt.cm.Greys, alpha=.8)
    ax.clabel(cs, colors='k')
    cs = ax.contour(xx, yy, Z, cmap=plt.cm.Greys, levels=[0, 0.5], linestyles='--', linewidths=2)
    ax.clabel(cs, colors='k')
    for k, xy0 in df[['x', 'y']].iterrows():
        x0, y0 = xy0.values
        plt.plot(x0, y0, markers[labels[k][0]], mec='k')

    markers = {0:'ws', 1:'w^'}    
    df = DataFrame(dict(x=xy2[:,0], y=xy2[:,1], label=labels2.flatten()))
    for k, xy0 in df[['x', 'y']].iterrows():
        x0, y0 = xy0.values
        plt.plot(x0, y0, markers[labels2[k][0]], mec='k', alpha=0.7)

    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    plt.grid(linestyle='--', alpha=0.5)
    plt.title(title)
    plt.show()

    
def example_plot_wo_contour(xy, labels, title, filename=None):
    # Shape
    c_shape = ['bs', 'r^']

    # 1. Point
    for k, (point, label) in enumerate(zip(xy, labels),1):
        x,y = point
        plt.plot(x, y, c_shape[label[0]], mec='k', markersize=12)
        plt.text(x+0.02, y-0.01, '$P_{}$'.format(k), size=15, \
                verticalalignment='top', horizontalalignment='left')

    plt.grid(linestyle='--', alpha=0.5)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title(title)
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

def plot_scatter_softmax(curr_W, curr_b, xy, labels):
    x1 = np.linspace(-0.1, 1.1)
    X, Y = np.meshgrid(x1,x1)
    Z1 = X * curr_W[0,0] + Y * curr_W[1,0] + curr_b[0]
    Z2 = X * curr_W[0,1] + Y * curr_W[1,1] + curr_b[1]
    Z = np.exp(Z1) / (np.exp(Z1) + np.exp(Z2))
    markers = ['bs', 'r^']
    plt.figure(figsize=(5,5))
    cs = plt.contourf(X, Y, Z, np.linspace(0, 1, 11), cmap=plt.cm.Greys, alpha=.8)
    plt.clabel(cs, colors='k')
    for k, xy0 in enumerate(xy):
        x0, y0 = xy0
        z1 = x0 * curr_W[0,0] + y0 * curr_W[1,0] + curr_b[0]
        z2 = x0 * curr_W[0,1] + y0 * curr_W[1,1] + curr_b[1]
        z = np.array([z1,z2])
        softmax_z = np.exp(z) / np.sum(np.exp(z))
        plt.plot(x0, y0, markers[labels[k][0]], mec='k', markersize=12)
        if labels[k][0]==0:
            plt.text(x0+0.02, y0-0.01, '$P_{}$\n({:1.1f},{:1.1f})'.format(k+1, softmax_z[0], softmax_z[1]), size=15, \
                 verticalalignment='top', horizontalalignment='left')
        else:
            plt.text(x0-0.08, y0-0.01, '$P_{}$\n({:1.1f},{:1.1f})'.format(k+1, softmax_z[0], softmax_z[1]), size=15, \
                 verticalalignment='top', horizontalalignment='left', color='w')
        
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()